/**
 * The client half of offset's editor bridge.
 *
 * The bridge is a Unix socket speaking newline-delimited JSON-RPC, and it
 * requires a token before it will dispatch anything: the socket can apply
 * edits, so an unauthenticated local socket would be a real vulnerability.
 * The token is written 0600 beside a descriptor file, and `readDescriptor`
 * performs exactly the same three-step dance the Python side documents —
 * find the descriptor, read the token from the path it names, connect.
 *
 * Reconnection is deliberate rather than eager. offset is a terminal program
 * the user starts and stops; an extension that retried in a tight loop would
 * spend the day failing to connect to something that is not running. One timer,
 * one attempt, and a status bar that says which state it is in.
 */

import * as fs from "fs";
import * as net from "net";
import * as os from "os";
import * as path from "path";
import { EventEmitter } from "events";

export interface Descriptor {
  version?: string;
  protocol?: string;
  /** "unix" or "tcp"; the bridge writes one or the other, never both. */
  transport?: string;
  /** The socket path when `transport` is "unix"; empty otherwise. */
  path?: string;
  host?: string;
  port?: number;
  token?: string;
  token_path?: string;
  pid?: number;
  started?: number;
  events?: string[];
  methods?: string[];
}

export interface StatusPayload {
  model?: string;
  state?: string;
  session?: string;
  workspace?: string;
  jobs?: Array<{ id: string; state: string; prompt?: string }>;
  turn?: { active?: boolean; tool?: string };
}

export interface ChangePayload {
  path: string;
  status: string;
  additions?: number;
  deletions?: number;
  diff?: string;
}

interface Pending {
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
  timer: NodeJS.Timeout;
}

/** Where offset keeps its state, honouring the same precedence the CLI uses. */
export function offsetHome(configured?: string): string {
  if (configured && configured.trim()) {
    return configured.trim();
  }
  const fromEnv = process.env.OFFSET_HOME;
  if (fromEnv && fromEnv.trim()) {
    return fromEnv.trim();
  }
  return path.join(os.homedir(), ".offset");
}

/**
 * Find a running bridge. Returns the descriptor with its token resolved, or a
 * reason it could not be read — never throws, because "offset is not running"
 * is the normal case and not an error worth a stack trace.
 */
export function readDescriptor(home: string): { descriptor?: Descriptor; problem?: string } {
  const descriptorPath = path.join(home, "bridge.json");
  if (!fs.existsSync(descriptorPath)) {
    return { problem: `no bridge descriptor at ${descriptorPath}; is offset running?` };
  }
  let raw: Descriptor;
  try {
    raw = JSON.parse(fs.readFileSync(descriptorPath, "utf8")) as Descriptor;
  } catch (err) {
    return { problem: `${descriptorPath}: ${(err as Error).message}` };
  }
  const tokenPath = raw.token_path ?? path.join(home, "bridge.token");
  try {
    raw.token = fs.readFileSync(tokenPath, "utf8").trim();
  } catch (err) {
    return { problem: `could not read the bridge token: ${(err as Error).message}` };
  }
  if (!raw.token) {
    return { problem: "the bridge token file is empty" };
  }
  return { descriptor: raw };
}

export type BridgeState = "offline" | "connecting" | "ready";

/**
 * A connected bridge session.
 *
 * Emits `state`, `event` (server-pushed notifications) and `closed`. Requests
 * are correlated by id with a deadline, so a bridge that stops answering
 * rejects its callers instead of leaving the UI waiting forever.
 */
export class BridgeClient extends EventEmitter {
  private socket?: net.Socket;
  private buffer = "";
  private nextId = 1;
  private readonly pending = new Map<number, Pending>();
  private closing = false;

  public state: BridgeState = "offline";
  public descriptor?: Descriptor;

  constructor(private readonly timeoutMs = 8000) {
    super();
  }

  private setState(next: BridgeState): void {
    if (this.state !== next) {
      this.state = next;
      this.emit("state", next);
    }
  }

  /** Connect and authenticate. Resolves once the bridge has accepted the token. */
  public async connect(home: string): Promise<void> {
    const { descriptor, problem } = readDescriptor(home);
    if (!descriptor) {
      this.setState("offline");
      throw new Error(problem ?? "no bridge descriptor");
    }
    this.descriptor = descriptor;
    this.setState("connecting");

    const { promise, resolve, reject } = Promise.withResolvers<void>();
    // The bridge writes `transport: "unix"` with a `path`, or `transport:
    // "tcp"` with host and port.  Reading the wrong field silently produced a
    // connection to port 0, whose failure carries no message at all.
    const socket = descriptor.transport === "unix" && descriptor.path
      ? net.createConnection(descriptor.path)
      : net.createConnection({
          host: descriptor.host || "127.0.0.1",
          port: descriptor.port ?? 0,
        });
    this.socket = socket;
    socket.setEncoding("utf8");

    const failed = (err: Error) => {
      socket.destroy();
      this.setState("offline");
      reject(err);
    };
    socket.once("error", failed);
    socket.once("connect", () => {
      socket.removeListener("error", failed);
      socket.on("error", (err) => this.emit("closed", (err as Error).message));
      socket.on("close", () => {
        if (!this.closing) {
          this.emit("closed", "the bridge closed the connection");
        }
        this.setState("offline");
        this.rejectAll(new Error("the bridge connection closed"));
      });
      socket.on("data", (chunk: string) => this.onData(chunk));
      resolve();
    });
    await promise;

    // `hello` is answered before authentication; every other method is refused
    // until the token has been accepted.
    await this.request("hello", { token: descriptor.token, client: "vscode" });
    this.setState("ready");
  }

  private onData(chunk: string): void {
    this.buffer += chunk;
    let index = this.buffer.indexOf("\n");
    while (index >= 0) {
      const line = this.buffer.slice(0, index).trim();
      this.buffer = this.buffer.slice(index + 1);
      if (line) {
        this.onLine(line);
      }
      index = this.buffer.indexOf("\n");
    }
  }

  private onLine(line: string): void {
    let message: Record<string, unknown>;
    try {
      message = JSON.parse(line) as Record<string, unknown>;
    } catch {
      return; // a frame we cannot parse is not worth killing the session over
    }
    const id = message.id;
    if (typeof id === "number" && this.pending.has(id)) {
      const waiter = this.pending.get(id)!;
      this.pending.delete(id);
      clearTimeout(waiter.timer);
      if (message.error) {
        const err = message.error as { message?: string; code?: number };
        waiter.reject(new Error(err.message ?? `bridge error ${err.code ?? "?"}`));
      } else {
        waiter.resolve(message.result);
      }
      return;
    }
    if (typeof message.method === "string") {
      this.emit("event", message.method, message.params ?? {});
    }
  }

  private rejectAll(reason: Error): void {
    for (const [, waiter] of this.pending) {
      clearTimeout(waiter.timer);
      waiter.reject(reason);
    }
    this.pending.clear();
  }

  /** Call a bridge method. Rejects on timeout rather than hanging the UI. */
  public request(method: string, params: Record<string, unknown> = {}): Promise<unknown> {
    const socket = this.socket;
    if (!socket || socket.destroyed) {
      return Promise.reject(new Error("not connected to offset"));
    }
    const id = this.nextId++;
    const frame = JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n";
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`${method} did not answer within ${this.timeoutMs}ms`));
      }, this.timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      socket.write(frame, (err) => {
        if (err) {
          clearTimeout(timer);
          this.pending.delete(id);
          reject(err);
        }
      });
    });
  }

  public async status(): Promise<StatusPayload> {
    return (await this.request("status")) as StatusPayload;
  }

  public async sessions(): Promise<Array<Record<string, unknown>>> {
    const reply = (await this.request("sessions")) as { sessions?: Array<Record<string, unknown>> };
    return reply?.sessions ?? [];
  }

  public async diff(): Promise<ChangePayload[]> {
    const reply = (await this.request("diff")) as { changes?: ChangePayload[] };
    return reply?.changes ?? [];
  }

  public dispose(): void {
    this.closing = true;
    this.rejectAll(new Error("the extension is shutting down"));
    this.socket?.destroy();
    this.socket = undefined;
    this.setState("offline");
  }
}
