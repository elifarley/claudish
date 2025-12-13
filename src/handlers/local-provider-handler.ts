/**
 * Local Provider Handler
 *
 * Handles requests to local OpenAI-compatible providers like Ollama, LM Studio, vLLM.
 * Uses the Provider Registry for configuration and shared OpenAI-compat utilities.
 */

import type { Context } from "hono";
import type { ModelHandler } from "./types.js";
import type { LocalProvider } from "../providers/provider-registry.js";
import { AdapterManager } from "../adapters/adapter-manager.js";
import { MiddlewareManager } from "../middleware/index.js";
import { transformOpenAIToClaude } from "../transform.js";
import { log, logStructured } from "../logger.js";
import {
  convertMessagesToOpenAI,
  convertToolsToOpenAI,
  filterIdentity,
  createStreamingResponseHandler,
  estimateTokens,
} from "./shared/openai-compat.js";

export class LocalProviderHandler implements ModelHandler {
  private provider: LocalProvider;
  private modelName: string;
  private adapterManager: AdapterManager;
  private middlewareManager: MiddlewareManager;
  private port: number;
  private healthChecked = false;
  private isHealthy = false;

  constructor(provider: LocalProvider, modelName: string, port: number) {
    this.provider = provider;
    this.modelName = modelName;
    this.port = port;
    this.adapterManager = new AdapterManager(modelName);
    this.middlewareManager = new MiddlewareManager();
    this.middlewareManager.initialize().catch((err) => {
      log(`[LocalProvider:${provider.name}] Middleware init error: ${err}`);
    });
  }

  /**
   * Check if the local provider is available
   */
  async checkHealth(): Promise<boolean> {
    if (this.healthChecked) return this.isHealthy;

    try {
      const healthUrl = `${this.provider.baseUrl}/api/tags`; // Ollama-specific health check
      const response = await fetch(healthUrl, {
        method: "GET",
        signal: AbortSignal.timeout(5000),
      });

      if (response.ok) {
        this.isHealthy = true;
        this.healthChecked = true;
        log(`[LocalProvider:${this.provider.name}] Health check passed`);
        return true;
      }
    } catch (e) {
      // Try alternative health check (generic OpenAI-compatible)
      try {
        const modelsUrl = `${this.provider.baseUrl}/v1/models`;
        const response = await fetch(modelsUrl, {
          method: "GET",
          signal: AbortSignal.timeout(5000),
        });
        if (response.ok) {
          this.isHealthy = true;
          this.healthChecked = true;
          log(`[LocalProvider:${this.provider.name}] Health check passed (v1/models)`);
          return true;
        }
      } catch (e2) {}
    }

    this.healthChecked = true;
    this.isHealthy = false;
    return false;
  }

  async handle(c: Context, payload: any): Promise<Response> {
    const target = this.modelName;

    logStructured(`LocalProvider Request`, {
      provider: this.provider.name,
      targetModel: target,
      originalModel: payload.model,
      baseUrl: this.provider.baseUrl,
    });

    // Health check on first request
    if (!this.healthChecked) {
      const healthy = await this.checkHealth();
      if (!healthy) {
        return this.errorResponse(c, "connection_error", this.getConnectionErrorMessage());
      }
    }

    // Transform request
    const { claudeRequest, droppedParams } = transformOpenAIToClaude(payload);
    const messages = convertMessagesToOpenAI(claudeRequest, target, filterIdentity);
    const tools = convertToolsToOpenAI(claudeRequest);

    // Check capability: strip tools if not supported
    const finalTools = this.provider.capabilities.supportsTools ? tools : [];
    if (tools.length > 0 && !this.provider.capabilities.supportsTools) {
      log(`[LocalProvider:${this.provider.name}] Tools stripped (not supported)`);
    }

    // Build OpenAI-compatible payload
    const openAIPayload: any = {
      model: target,
      messages,
      temperature: claudeRequest.temperature ?? 1,
      stream: this.provider.capabilities.supportsStreaming,
      max_tokens: claudeRequest.max_tokens,
      tools: finalTools.length > 0 ? finalTools : undefined,
      stream_options: this.provider.capabilities.supportsStreaming ? { include_usage: true } : undefined,
    };

    // Handle tool choice
    if (claudeRequest.tool_choice && finalTools.length > 0) {
      const { type, name } = claudeRequest.tool_choice;
      if (type === "tool" && name) {
        openAIPayload.tool_choice = { type: "function", function: { name } };
      } else if (type === "auto" || type === "none") {
        openAIPayload.tool_choice = type;
      }
    }

    // Apply adapter transformations
    const adapter = this.adapterManager.getAdapter();
    if (typeof adapter.reset === "function") adapter.reset();
    adapter.prepareRequest(openAIPayload, claudeRequest);

    // Apply middleware
    await this.middlewareManager.beforeRequest({
      modelId: target,
      messages,
      tools: finalTools,
      stream: openAIPayload.stream,
    });

    // Make request to local provider
    const apiUrl = `${this.provider.baseUrl}${this.provider.apiPath}`;

    try {
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(openAIPayload),
      });

      if (!response.ok) {
        const errorBody = await response.text();
        return this.handleErrorResponse(c, response.status, errorBody);
      }

      if (droppedParams.length > 0) {
        c.header("X-Dropped-Params", droppedParams.join(", "));
      }

      // Handle streaming response
      if (openAIPayload.stream) {
        return createStreamingResponseHandler(
          c,
          response,
          adapter,
          target,
          this.middlewareManager
        );
      }

      // Handle non-streaming response (shouldn't normally happen)
      const data = await response.json();
      return c.json(data);
    } catch (error: any) {
      if (error.code === "ECONNREFUSED" || error.cause?.code === "ECONNREFUSED") {
        return this.errorResponse(c, "connection_error", this.getConnectionErrorMessage());
      }
      throw error;
    }
  }

  private handleErrorResponse(c: Context, status: number, errorBody: string): Response {
    // Parse error and provide helpful messages
    try {
      const parsed = JSON.parse(errorBody);
      const errorMsg = parsed.error?.message || parsed.error || errorBody;

      // Model not found
      if (errorMsg.includes("model") && (errorMsg.includes("not found") || errorMsg.includes("does not exist"))) {
        return this.errorResponse(
          c,
          "model_not_found",
          `Model '${this.modelName}' not found. ${this.getModelPullHint()}`
        );
      }

      // Model doesn't support tools - provide helpful message
      if (errorMsg.includes("does not support tools") || errorMsg.includes("tool") && errorMsg.includes("not supported")) {
        return this.errorResponse(
          c,
          "capability_error",
          `Model '${this.modelName}' does not support tool/function calling. Claude Code requires tool support for most operations. Try a model that supports tools (e.g., llama3.2, mistral, qwen2.5).`,
          400
        );
      }

      return this.errorResponse(c, "api_error", errorMsg, status);
    } catch {
      return this.errorResponse(c, "api_error", errorBody, status);
    }
  }

  private errorResponse(c: Context, type: string, message: string, status: number = 503): Response {
    return c.json(
      {
        error: {
          type,
          message,
        },
      },
      status as any
    );
  }

  private getConnectionErrorMessage(): string {
    switch (this.provider.name) {
      case "ollama":
        return `Cannot connect to Ollama at ${this.provider.baseUrl}. Make sure Ollama is running with: ollama serve`;
      case "lmstudio":
        return `Cannot connect to LM Studio at ${this.provider.baseUrl}. Make sure LM Studio server is running.`;
      case "vllm":
        return `Cannot connect to vLLM at ${this.provider.baseUrl}. Make sure vLLM server is running.`;
      default:
        return `Cannot connect to ${this.provider.name} at ${this.provider.baseUrl}. Make sure the server is running.`;
    }
  }

  private getModelPullHint(): string {
    switch (this.provider.name) {
      case "ollama":
        return `Pull it with: ollama pull ${this.modelName}`;
      default:
        return "Make sure the model is available on the server.";
    }
  }

  async shutdown(): Promise<void> {}
}
