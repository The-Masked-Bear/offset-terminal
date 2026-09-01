/**
 * `@offset/client` - drive offset from TypeScript.
 *
 * offset's agent is Python and stays Python: thirty-odd thousand lines and a
 * thousand tests do not get rewritten for a language preference. What belongs
 * in TypeScript is everything on the *other* side of the socket - VS Code,
 * JetBrains, a Node script, a web front end - and this is that half.
 *
 * ```ts
 * import { OffsetClient } from "@offset/client";
 *
 * const offset = new OffsetClient({ name: "my-editor" });
 * await offset.connect();
 *
 * offset.on("tool.started", ({ name }) => console.log("running", name));
 *
 * const { text } = await offset.prompt("add a test for the parser");
 * console.log(text);
 * ```
 *
 * Zero runtime dependencies, matching the Python side's own policy: Node
 * built-ins only.
 */

export { OffsetClient } from "./client.js";
export type { ClientOptions } from "./client.js";
export { describe, discover, offsetHome } from "./discover.js";
export type { Discovery } from "./discover.js";
export {
  BRIDGE_VERSION,
  DESCRIPTOR_NAME,
  EVENTS,
  INTERNAL_ERROR,
  INVALID_PARAMS,
  METHOD_NOT_FOUND,
  METHODS,
  PROTOCOL,
  RpcError,
  SOCKET_NAME,
  TOKEN_NAME,
  UNAUTHENTICATED,
} from "./protocol.js";
export type {
  ChangePayload,
  ChangesPayload,
  ClientState,
  Connection,
  Descriptor,
  EventName,
  JobPayload,
  MethodName,
  PromptPayload,
  SessionPayload,
  StatusPayload,
} from "./protocol.js";
