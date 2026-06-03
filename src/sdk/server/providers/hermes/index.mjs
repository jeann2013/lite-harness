// Hermes provider: drives @openai/agents against a local OpenAI-compatible
// endpoint (Ollama, vLLM, LiteLLM, etc.) and maps its run-stream events to
// the canonical wire using the same transformation as the codex provider.
import { Agent, run, setDefaultOpenAIClient, setOpenAIAPI, setTracingDisabled } from "@openai/agents";
import OpenAI from "openai";
import { eventToFrames } from "../codex/transformation.mjs";

export const id = "hermes";
export const aliases = ["nous-hermes", "hermes-agent"];

let configured = false;
function configure(env) {
  if (configured) return;
  configured = true;
  const base = env.HERMES_API_BASE;
  if (!base) throw new Error("HERMES_API_BASE is required for the hermes provider");
  const baseURL = base.replace(/\/+$/, "");
  const apiKey = env.HERMES_API_KEY || "ollama";
  setTracingDisabled(true);
  setDefaultOpenAIClient(new OpenAI({ baseURL: baseURL.endsWith("/v1") ? baseURL : `${baseURL}/v1`, apiKey }));
  setOpenAIAPI("chat_completions");
}

export function createRuntime({ model, env = process.env, diagnostics = () => {} }) {
  configure(env);
  let currentModel = model || env.HERMES_DEFAULT_MODEL;
  let aborter = null;

  return {
    get model() {
      return currentModel;
    },
    setModel(next) {
      if (next) currentModel = next;
    },
    setPermissionMode() {},
    interrupt() {
      aborter?.abort();
    },
    async *runTurn({ prompt, session }) {
      aborter = new AbortController();
      const agent = new Agent({
        name: "hermes",
        model: currentModel,
        instructions: "You are a coding agent.",
      });
      const streamed = await run(agent, prompt, { stream: true, signal: aborter.signal });
      try {
        for await (const event of streamed) {
          for (const frame of eventToFrames(event, { sessionId: session.sessionId, model: currentModel })) {
            yield frame;
          }
        }
      } catch (err) {
        if (aborter.signal.aborted) return;
        diagnostics(`hermes runtime error: ${err?.message ?? err}\n`);
        throw err;
      } finally {
        aborter = null;
      }
    },
  };
}
