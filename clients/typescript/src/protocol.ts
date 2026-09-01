/**
 * The bridge protocol, in types.
 *
 * This file is the TypeScript half of `offset/core/bridge.py` and is the only
 * place the wire format is written down on this side. Everything it declares
 * has a named counterpart there, so when the two disagree the disagreement is
 * a diff in one file rather than a mystery spread across a client.
 *
 * The protocol is newline-delimited JSON-RPC 2.0 over a socket - a Unix socket
 * for an editor on the same machine, TCP for a daemon reached across an SSH
 * hop. A token is required before anything is dispatched, because the socket
 * can apply edits and run tools: an unauthenticated one would be a real
 * vulnerability rather than a theoretical one.
 */

/** JSON-RPC version. The bridge speaks 2.0 and rejects anything else. */
export const PROTOCOL = "2.0" as const;

/** Descriptor layout version, bumped when the discovery file changes shape. */
export const BRIDGE_VERSION = "1" as const;

/** Filenames the bridge publishes inside `$OFFSET_HOME`. */
export const SOCKET_NAME = "bridge.sock" as const;
export const TOKEN_NAME = "bridge.token" as const;
export const DESCRIPTOR_NAME = "bridge.json" as const;

/**
 * Application error code for "you have not authenticated".
 *
 * Outside the JSON-RPC reserved range, as the spec requires. A client keys its
 * "not authorised" state on exactly this rather than on message text.
 */
export const UNAUTHENTICATED = -32001 as const;

export const METHOD_NOT_FOUND = -32601 as const;
export const INVALID_PARAMS = -32602 as const;
export const INTERNAL_ERROR = -32603 as const;

/**
 * Every notification the bridge will ever push.
 *
 * A closed set on purpose: a client written against "whatever I happened to
 * observe" breaks the first time a quiet event fires.
 */
export const EVENTS = [
  "agent.started",
  "agent.finished",
  "tool.started",
  "tool.finished",
  "edit.applied",
  "job.state",
] as const;

export type EventName = (typeof EVENTS)[number];

/**
 * Methods callable once authenticated.
 *
 * `hello` is deliberately absent: it is answered *before* authentication and
 * so is not in the dispatch table on either side.
 */
export const METHODS = [
  "status",
  "sessions",
  "diff",
  "apply_edit",
  "cancel",
  "prompt",
] as const;

export type MethodName = (typeof METHODS)[number];

/** How to reach a running bridge. Written `0o600` beside the token. */
export interface Descriptor {
  version: string;
  protocol: string;
  /** `unix` names `path`; `tcp` names `host` and `port`. Never both. */
  transport: "unix" | "tcp";
  path?: string;
  host?: string;
  port?: number;
  token_path?: string;
  pid?: number;
  started?: number;
  events?: string[];
  methods?: string[];
}

/** A descriptor with its token already read from the path it names. */
export interface Connection extends Descriptor {
  token: string;
}

/** One background job, as `status.jobs` and the `job.state` event carry it. */
export interface JobPayload {
  id: string;
  state: string;
  [key: string]: unknown;
}

/**
 * What `status` answers.
 *
 * Field names are the bridge's, not a tidied version of them: `Bridge._status`
 * is the authority and a client that renames things here only moves the
 * mismatch somewhere harder to find.
 */
export interface StatusPayload {
  model?: string;
  workspace?: string;
  session?: string;
  session_path?: string;
  entries?: number;
  /** `running` while a turn is in flight, otherwise `idle`. */
  state?: string;
  jobs?: JobPayload[];
  running_jobs?: number;
  clients?: number;
  dropped?: number;
  pid?: number;
  version?: string;
  started?: number;
  [key: string]: unknown;
}

/**
 * One file the agent has changed but not yet committed.
 *
 * `original` and `current` are both sent so an editor can show a real
 * side-by-side diff without re-reading the working tree and risking a view of
 * two different moments.  When `truncated` is set the file was binary or too
 * large, all three text fields are empty, and only the counts are meaningful -
 * rendering an empty diff instead would read as "no change".
 */
export interface ChangePayload {
  path: string;
  status: string;
  additions: number;
  deletions: number;
  diff: string;
  original: string;
  current: string;
  truncated: boolean;
}

export interface ChangesPayload {
  changes: ChangePayload[];
  [key: string]: unknown;
}

export interface SessionPayload {
  id: string;
  [key: string]: unknown;
}

export interface PromptPayload {
  ok: boolean;
  text?: string;
  [key: string]: unknown;
}

/** A JSON-RPC failure, carried as an exception rather than a return value. */
export class RpcError extends Error {
  readonly code: number;
  readonly data: unknown;

  constructor(code: number, message: string, data?: unknown) {
    super(message);
    this.name = "RpcError";
    this.code = code;
    this.data = data;
  }

  /** Whether this is the bridge saying the token was wrong or missing. */
  get unauthenticated(): boolean {
    return this.code === UNAUTHENTICATED;
  }
}

export type ClientState = "offline" | "connecting" | "ready";
