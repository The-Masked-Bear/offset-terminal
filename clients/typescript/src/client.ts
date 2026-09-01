/**
 * A connected offset session.
 *
 * One socket, newline-delimited JSON-RPC, and a `hello` that must succeed
 * before anything else is sent. Every method here is a thin, typed wrapper
 * over `request`, which is public: a bridge that grows a method should be
 * usable from this client the same day, without waiting for a release.
 *
 * Reconnection is deliberate rather than eager. offset is a program somebody
 * starts and stops; a client that retried in a tight loop would spend the day
 * failing to reach something that is not running. One timer, one attempt, and
 * a state a caller can render.
 */

import * as net from "node:net";
import { EventEmitter } from "node:events";

import { describe, discover } from "./discover.js";
import {
  ChangePayload,
  ChangesPayload,
  ClientState,
  Connection,
  EventName,
  PromptPayload,
  RpcError,
  SessionPayload,
  StatusPayload,
} from "./protocol.js";

export interface ClientOptions {
  /** Overrides `$OFFSET_HOME`; mostly for tests and multi-checkout setups. */
  home?: string;
  /** Identifies this client in the bridge's logs. */
  name?: string;
  /** Milliseconds before an unanswered request is abandoned. */
  timeout?: number;
  /** Milliseconds between reconnection attempts. 0 disables reconnecting. */
  retry?: number;
}

interface Pending {
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
  timer: NodeJS.Timeout;
}

const DEFAULT_TIMEOUT = 30_000;
const DEFAULT_RETRY = 5_000;

/**
 * A frame larger than this is refused rather than buffered.
 *
 * The socket is authenticated, but a client that grows its buffer without
 * limit turns one confused peer into an out-of-memory kill.
 */
const MAX_FRAME = 32 * 1024 * 1024;

export declare interface OffsetClient {
  on(event: "state", listener: (state: ClientState) => void): this;
  on(event: "closed", listener: (why: string) => void): this;
  on(event: EventName, listener: (params: Record<string, unknown>) => void): this;
  on(event: string, listener: (...args: never[]) => void): this;
}

export class OffsetClient extends EventEmitter {
  private socket?: net.Socket;
  private buffer = "";
  private nextId = 1;
  private readonly pending = new Map<number, Pending>();
  private closing = false;
  private retryTimer?: NodeJS.Timeout;

  state: ClientState = "offline";
  connection?: Connection;
  /** Why the last attempt failed, in words worth showing a user. */
  problem = "";

  constructor(private readonly options: ClientOptions = {}) {
    super();
  }

  get ready(): boolean {
    return this.state === "ready";
  }

  /** Where this client is connected, or would connect. */
  get address(): string {
    return this.connection ? describe(this.connection) : "";
  }

  private setState(next: ClientState): void {
    if (this.state === next) return;
    this.state = next;
    this.emit("state", next);
  }

  /**
   * Find offset, connect, and authenticate.
   *
   * Resolves once `hello` has been accepted; until then no other method will
   * be dispatched by the bridge, so resolving earlier would hand back a client
   * whose every call fails.
   */
  async connect(home?: string): Promise<void> {
    this.closing = false;
    // An override rather than a fixed option: an editor's "offset home"
    // setting can change between attempts, and a reconnect must use the
    // current one rather than whatever was true when the client was built.
    if (home !== undefined) this.options.home = home;
    const found = discover(this.options.home);
    if (!found.connection) {
      this.problem = found.problem ?? "offset is not running";
      this.setState("offline");
      throw new Error(this.problem);
    }
    this.connection = found.connection;
    this.setState("connecting");

    const socket = await this.open(found.connection);
    this.socket = socket;
    socket.setEncoding("utf8");
    socket.on("data", (chunk: string) => this.onData(chunk));
    socket.on("error", (error: Error) => this.fail(error.message));
    socket.on("close", () => this.fail("the bridge closed the connection"));

    await this.request("hello", {
      token: found.connection.token,
      client: this.options.name ?? "offset-client",
    });
    this.setState("ready");
  }

  private open(connection: Connection): Promise<net.Socket> {
    return new Promise((resolve, reject) => {
      const socket =
        connection.transport === "unix"
          ? net.createConnection({ path: String(connection.path) })
          : net.createConnection({
              host: connection.host || "127.0.0.1",
              port: Number(connection.port),
            });
      const failed = (error: Error) => {
        socket.destroy();
        this.problem = error.message;
        this.setState("offline");
        reject(error);
      };
      socket.once("error", failed);
      socket.once("connect", () => {
        socket.off("error", failed);
        resolve(socket);
      });
    });
  }

  /** Send one request and wait for its reply. */
  request<T = unknown>(method: string, params: Record<string, unknown> = {}): Promise<T> {
    const socket = this.socket;
    if (!socket || socket.destroyed) {
      return Promise.reject(new Error("not connected to offset"));
    }
    const id = this.nextId++;
    const frame = JSON.stringify({ jsonrpc: "2.0", id, method, params });

    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`${method} did not answer within ${this.timeoutMs}ms`));
      }, this.timeoutMs);
      // `unref` so a pending request cannot hold a CLI process open.
      timer.unref?.();
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, timer });
      socket.write(frame + "\n");
    });
  }

  private get timeoutMs(): number {
    return this.options.timeout ?? DEFAULT_TIMEOUT;
  }

  private onData(chunk: string): void {
    this.buffer += chunk;
    if (this.buffer.length > MAX_FRAME) {
      this.fail(`a single frame exceeded ${MAX_FRAME} bytes`);
      return;
    }
    let index = this.buffer.indexOf("\n");
    while (index >= 0) {
      const line = this.buffer.slice(0, index).trim();
      this.buffer = this.buffer.slice(index + 1);
      if (line) this.dispatch(line);
      index = this.buffer.indexOf("\n");
    }
  }

  private dispatch(line: string): void {
    let message: Record<string, unknown>;
    try {
      message = JSON.parse(line) as Record<string, unknown>;
    } catch {
      return; // a frame we cannot read is not a reason to drop the connection
    }

    const id = message.id;
    if (typeof id === "number") {
      const waiting = this.pending.get(id);
      if (!waiting) return;
      this.pending.delete(id);
      clearTimeout(waiting.timer);
      const failure = message.error as { code?: number; message?: string; data?: unknown } | undefined;
      if (failure) {
        waiting.reject(
          new RpcError(failure.code ?? 0, failure.message ?? "the bridge refused", failure.data),
        );
      } else {
        waiting.resolve(message.result);
      }
      return;
    }

    // No id: a notification. These are the events the agent pushes.
    const method = message.method;
    if (typeof method === "string") {
      this.emit(method, (message.params as Record<string, unknown>) ?? {});
    }
  }

  private fail(why: string): void {
    this.problem = why;
    const socket = this.socket;
    this.socket = undefined;
    socket?.destroy();

    for (const [, waiting] of this.pending) {
      clearTimeout(waiting.timer);
      waiting.reject(new Error(why));
    }
    this.pending.clear();

    this.setState("offline");
    this.emit("closed", why);
    if (!this.closing) this.scheduleRetry();
  }

  private scheduleRetry(): void {
    const delay = this.options.retry ?? DEFAULT_RETRY;
    if (!delay || this.retryTimer) return;
    this.retryTimer = setTimeout(() => {
      this.retryTimer = undefined;
      if (this.closing) return;
      void this.connect().catch(() => {
        // Already recorded in `problem`, and a retry that fails schedules the
        // next one through `fail`. Nothing further to do here.
      });
    }, delay);
    this.retryTimer.unref?.();
  }

  /** Stop, and stay stopped: no reconnection follows a deliberate close. */
  dispose(): void {
    this.closing = true;
    if (this.retryTimer) {
      clearTimeout(this.retryTimer);
      this.retryTimer = undefined;
    }
    const socket = this.socket;
    this.socket = undefined;
    socket?.destroy();
    for (const [, waiting] of this.pending) {
      clearTimeout(waiting.timer);
      waiting.reject(new Error("the client was disposed"));
    }
    this.pending.clear();
    this.setState("offline");
  }

  // -- the methods ----------------------------------------------------------

  status(): Promise<StatusPayload> {
    return this.request<StatusPayload>("status");
  }

  async sessions(): Promise<SessionPayload[]> {
    const reply = await this.request<SessionPayload[] | { sessions?: SessionPayload[] }>("sessions");
    return Array.isArray(reply) ? reply : (reply?.sessions ?? []);
  }

  async diff(): Promise<ChangePayload[]> {
    const reply = await this.request<ChangesPayload>("diff");
    return reply?.changes ?? [];
  }

  applyEdit(target: string, text: string): Promise<unknown> {
    return this.request("apply_edit", { target, text });
  }

  cancel(): Promise<unknown> {
    return this.request("cancel");
  }

  /** Run a real turn. Resolves when the agent has finished, not when it starts. */
  prompt(text: string, selection?: unknown): Promise<PromptPayload> {
    return this.request<PromptPayload>("prompt", selection === undefined ? { text } : { text, selection });
  }
}
