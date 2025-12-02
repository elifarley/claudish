import { BaseModelAdapter, AdapterResult } from "./base-adapter";
import { log } from "../logger";

export class QwenAdapter extends BaseModelAdapter {
  processTextContent(textContent: string, accumulatedText: string): AdapterResult {
    // Qwen models return standard content
    // However, some newer models might wrap thinking in <think> tags which we might want to handle
    // For now, we pass through as is, similar to OpenAI
    return {
      cleanedText: textContent,
      extractedToolCalls: [],
      wasTransformed: false,
    };
  }

  /**
   * Handle request preparation - specifically for mapping reasoning parameters
   */
  override prepareRequest(request: any, originalRequest: any): any {
    if (originalRequest.thinking) {
      const { budget_tokens } = originalRequest.thinking;

      // Qwen specific parameters
      request.enable_thinking = true;
      request.thinking_budget = budget_tokens;

      log(
        `[QwenAdapter] Mapped budget ${budget_tokens} -> enable_thinking: true, thinking_budget: ${budget_tokens}`
      );

      // Cleanup: Remove raw thinking object
      delete request.thinking;
    }

    return request;
  }

  shouldHandle(modelId: string): boolean {
    return modelId.includes("qwen") || modelId.includes("alibaba");
  }

  getName(): string {
    return "QwenAdapter";
  }
}
