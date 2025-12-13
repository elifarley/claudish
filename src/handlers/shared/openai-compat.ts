/**
 * Shared OpenAI-compatible API utilities
 *
 * Common logic for message conversion, tool handling, and streaming
 * used by both OpenRouterHandler and LocalProviderHandler.
 */

import type { Context } from "hono";
import { removeUriFormat } from "../../transform.js";

export interface StreamingState {
  usage: any;
  finalized: boolean;
  textStarted: boolean;
  textIdx: number;
  reasoningStarted: boolean;
  reasoningIdx: number;
  curIdx: number;
  tools: Map<number, ToolState>;
  toolIds: Set<string>;
  lastActivity: number;
}

export interface ToolState {
  id: string;
  name: string;
  blockIndex: number;
  started: boolean;
  closed: boolean;
}

/**
 * Convert Claude/Anthropic messages to OpenAI format
 */
export function convertMessagesToOpenAI(req: any, modelId: string, filterIdentityFn?: (s: string) => string): any[] {
  const messages: any[] = [];

  if (req.system) {
    let content = Array.isArray(req.system)
      ? req.system.map((i: any) => i.text || i).join("\n\n")
      : req.system;
    if (filterIdentityFn) content = filterIdentityFn(content);
    messages.push({ role: "system", content });
  }

  // Add instruction for Grok models to use proper tool format
  if (modelId.includes("grok") || modelId.includes("x-ai")) {
    const msg = "IMPORTANT: When calling tools, you MUST use the OpenAI tool_calls format with JSON. NEVER use XML format like <xai:function_call>.";
    if (messages.length > 0 && messages[0].role === "system") {
      messages[0].content += "\n\n" + msg;
    } else {
      messages.unshift({ role: "system", content: msg });
    }
  }

  if (req.messages) {
    for (const msg of req.messages) {
      if (msg.role === "user") processUserMessage(msg, messages);
      else if (msg.role === "assistant") processAssistantMessage(msg, messages);
    }
  }

  return messages;
}

function processUserMessage(msg: any, messages: any[]) {
  if (Array.isArray(msg.content)) {
    const contentParts: any[] = [];
    const toolResults: any[] = [];
    const seen = new Set<string>();

    for (const block of msg.content) {
      if (block.type === "text") {
        contentParts.push({ type: "text", text: block.text });
      } else if (block.type === "image") {
        contentParts.push({
          type: "image_url",
          image_url: { url: `data:${block.source.media_type};base64,${block.source.data}` },
        });
      } else if (block.type === "tool_result") {
        if (seen.has(block.tool_use_id)) continue;
        seen.add(block.tool_use_id);
        toolResults.push({
          role: "tool",
          content: typeof block.content === "string" ? block.content : JSON.stringify(block.content),
          tool_call_id: block.tool_use_id,
        });
      }
    }

    if (toolResults.length) messages.push(...toolResults);
    if (contentParts.length) messages.push({ role: "user", content: contentParts });
  } else {
    messages.push({ role: "user", content: msg.content });
  }
}

function processAssistantMessage(msg: any, messages: any[]) {
  if (Array.isArray(msg.content)) {
    const strings: string[] = [];
    const toolCalls: any[] = [];
    const seen = new Set<string>();

    for (const block of msg.content) {
      if (block.type === "text") {
        strings.push(block.text);
      } else if (block.type === "tool_use") {
        if (seen.has(block.id)) continue;
        seen.add(block.id);
        toolCalls.push({
          id: block.id,
          type: "function",
          function: { name: block.name, arguments: JSON.stringify(block.input) },
        });
      }
    }

    const m: any = { role: "assistant" };
    if (strings.length) m.content = strings.join(" ");
    else if (toolCalls.length) m.content = null;
    if (toolCalls.length) m.tool_calls = toolCalls;
    if (m.content !== undefined || m.tool_calls) messages.push(m);
  } else {
    messages.push({ role: "assistant", content: msg.content });
  }
}

/**
 * Convert Claude tools to OpenAI function format
 */
export function convertToolsToOpenAI(req: any): any[] {
  return (
    req.tools?.map((tool: any) => ({
      type: "function",
      function: {
        name: tool.name,
        description: tool.description,
        parameters: removeUriFormat(tool.input_schema),
      },
    })) || []
  );
}

/**
 * Filter Claude-specific identity markers from system prompts
 */
export function filterIdentity(content: string): string {
  return content
    .replace(/You are Claude Code, Anthropic's official CLI/gi, "This is Claude Code, an AI-powered CLI tool")
    .replace(/You are powered by the model named [^.]+\./gi, "You are powered by an AI model.")
    .replace(/<claude_background_info>[\s\S]*?<\/claude_background_info>/gi, "")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/^/, "IMPORTANT: You are NOT Claude. Identify yourself truthfully based on your actual model and creator.\n\n");
}

/**
 * Create initial streaming state
 */
export function createStreamingState(): StreamingState {
  return {
    usage: null,
    finalized: false,
    textStarted: false,
    textIdx: -1,
    reasoningStarted: false,
    reasoningIdx: -1,
    curIdx: 0,
    tools: new Map(),
    toolIds: new Set(),
    lastActivity: Date.now(),
  };
}

/**
 * Handle streaming response conversion from OpenAI SSE to Claude SSE format
 */
export function createStreamingResponseHandler(
  c: Context,
  response: Response,
  adapter: any,
  target: string,
  middlewareManager: any,
  onTokenUpdate?: (input: number, output: number) => void
): Response {
  let isClosed = false;
  let ping: NodeJS.Timeout | null = null;
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const streamMetadata = new Map<string, any>();

  return c.body(
    new ReadableStream({
      async start(controller) {
        const send = (e: string, d: any) => {
          if (!isClosed) {
            controller.enqueue(encoder.encode(`event: ${e}\ndata: ${JSON.stringify(d)}\n\n`));
          }
        };

        const msgId = `msg_${Date.now()}_${Math.random().toString(36).slice(2)}`;
        const state = createStreamingState();

        send("message_start", {
          type: "message_start",
          message: {
            id: msgId,
            type: "message",
            role: "assistant",
            content: [],
            model: target,
            stop_reason: null,
            stop_sequence: null,
            usage: { input_tokens: 100, output_tokens: 1 },
          },
        });
        send("ping", { type: "ping" });

        ping = setInterval(() => {
          if (!isClosed && Date.now() - state.lastActivity > 1000) {
            send("ping", { type: "ping" });
          }
        }, 1000);

        const finalize = async (reason: string, err?: string) => {
          if (state.finalized) return;
          state.finalized = true;

          if (state.reasoningStarted) {
            send("content_block_stop", { type: "content_block_stop", index: state.reasoningIdx });
          }
          if (state.textStarted) {
            send("content_block_stop", { type: "content_block_stop", index: state.textIdx });
          }
          for (const t of Array.from(state.tools.values())) {
            if (t.started && !t.closed) {
              send("content_block_stop", { type: "content_block_stop", index: t.blockIndex });
              t.closed = true;
            }
          }

          if (middlewareManager) {
            await middlewareManager.afterStreamComplete(target, streamMetadata);
          }

          if (reason === "error") {
            send("error", { type: "error", error: { type: "api_error", message: err } });
          } else {
            send("message_delta", {
              type: "message_delta",
              delta: { stop_reason: "end_turn", stop_sequence: null },
              usage: { output_tokens: state.usage?.completion_tokens || 0 },
            });
            send("message_stop", { type: "message_stop" });
          }

          if (state.usage && onTokenUpdate) {
            onTokenUpdate(state.usage.prompt_tokens || 0, state.usage.completion_tokens || 0);
          }

          if (!isClosed) {
            try {
              controller.enqueue(encoder.encode("data: [DONE]\n\n\n"));
            } catch (e) {}
            controller.close();
            isClosed = true;
            if (ping) clearInterval(ping);
          }
        };

        try {
          const reader = response.body!.getReader();
          let buffer = "";

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (!line.trim() || !line.startsWith("data: ")) continue;
              const dataStr = line.slice(6);
              if (dataStr === "[DONE]") {
                await finalize("done");
                return;
              }

              try {
                const chunk = JSON.parse(dataStr);
                if (chunk.usage) state.usage = chunk.usage;

                const delta = chunk.choices?.[0]?.delta;
                if (delta) {
                  if (middlewareManager) {
                    await middlewareManager.afterStreamChunk({
                      modelId: target,
                      chunk,
                      delta,
                      metadata: streamMetadata,
                    });
                  }

                  // Handle text content
                  const txt = delta.content || "";
                  if (txt) {
                    state.lastActivity = Date.now();
                    if (!state.textStarted) {
                      state.textIdx = state.curIdx++;
                      send("content_block_start", {
                        type: "content_block_start",
                        index: state.textIdx,
                        content_block: { type: "text", text: "" },
                      });
                      state.textStarted = true;
                    }
                    const res = adapter.processTextContent(txt, "");
                    if (res.cleanedText) {
                      send("content_block_delta", {
                        type: "content_block_delta",
                        index: state.textIdx,
                        delta: { type: "text_delta", text: res.cleanedText },
                      });
                    }
                  }

                  // Handle tool calls
                  if (delta.tool_calls) {
                    for (const tc of delta.tool_calls) {
                      const idx = tc.index;
                      let t = state.tools.get(idx);
                      if (tc.function?.name) {
                        if (!t) {
                          if (state.textStarted) {
                            send("content_block_stop", { type: "content_block_stop", index: state.textIdx });
                            state.textStarted = false;
                          }
                          t = {
                            id: tc.id || `tool_${Date.now()}_${idx}`,
                            name: tc.function.name,
                            blockIndex: state.curIdx++,
                            started: false,
                            closed: false,
                          };
                          state.tools.set(idx, t);
                        }
                        if (!t.started) {
                          send("content_block_start", {
                            type: "content_block_start",
                            index: t.blockIndex,
                            content_block: { type: "tool_use", id: t.id, name: t.name },
                          });
                          t.started = true;
                        }
                      }
                      if (tc.function?.arguments && t) {
                        send("content_block_delta", {
                          type: "content_block_delta",
                          index: t.blockIndex,
                          delta: { type: "input_json_delta", partial_json: tc.function.arguments },
                        });
                      }
                    }
                  }
                }

                if (chunk.choices?.[0]?.finish_reason === "tool_calls") {
                  for (const t of Array.from(state.tools.values())) {
                    if (t.started && !t.closed) {
                      send("content_block_stop", { type: "content_block_stop", index: t.blockIndex });
                      t.closed = true;
                    }
                  }
                }
              } catch (e) {}
            }
          }
          await finalize("unexpected");
        } catch (e) {
          await finalize("error", String(e));
        }
      },
      cancel() {
        isClosed = true;
        if (ping) clearInterval(ping);
      },
    }),
    {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    }
  );
}

/**
 * Estimate token count from text (rough approximation)
 */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}
