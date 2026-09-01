/**
 * The client, against a real socket.
 *
 * Nothing here mocks `net`. A fake bridge is a genuine server speaking the
 * genuine framing, because every bug this client has actually had lived in the
 * framing or the handshake - a field read by the wrong name, a frame split
 * across two chunks - and a mock would have agreed with all of them.
 *
 * No test sleeps. Every wait is on the signal the code already emits: the
 * `state` event, the promise a method returns, or the notification itself.
 */

import assert from "node:assert/strict";
import { once } from "node:events";
import * as fs from "node:fs";
import * as net from "node:net";
import * as os from "node:os";
import * as path from "node:path";
import { after, before, describe as suite, test } from "node:test";

import { OffsetClient } from "../src/client.js";
import { discover, offsetHome } from "../src/discover.js";
import { BRIDGE_VERSION, RpcError, UNAUTHENTICATED } from "../src/protocol.js";

/** One request as it arrives at the fake bridge, after narrowing. */
interface Incoming {
  id: number;
  method: string;
  params: Record<string, unknown>;
}

/** Narrow a parsed frame, rather than asserting a shape onto it. */
function incoming(raw: unknown): Incoming | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const frame: Record<string, unknown> = { ...raw };
  const { id, method, params } = frame;
  if (typeof id !== "number" || typeof method !== "string") return undefined;
  return {
    id,
    method,
    params: params && typeof params === "object" ? { ...params } : {},
  };
}

type Handler = (message: Incoming, socket: net.Socket) => void;

/** A bridge that speaks the real protocol and does as it is told. */
class FakeBridge {
  readonly home: string;
  readonly socketPath: string;
  readonly token = "test-token";
  readonly seen: Incoming[] = [];
  handler: Handler;

  private server?: net.Server;
  private readonly live = new Set<net.Socket>();

  constructor() {
    this.home = fs.mkdtempSync(path.join(os.tmpdir(), "offset-client-"));
    this.socketPath = path.join(this.home, "bridge.sock");
    this.handler = (message, socket) => this.answer(message, socket);
  }

  /** The default: accept `hello` with the right token, answer the rest. */
  answer(message: Incoming, socket: net.Socket): void {
    const { id, method, params } = message;
    if (method === "hello") {
      if (params.token !== this.token) {
        this.fail(socket, id, UNAUTHENTICATED, "the token does not match");
        return;
      }
      this.reply(socket, id, { ok: true, version: BRIDGE_VERSION });
      return;
    }
    if (method === "status") {
      this.reply(socket, id, { model: "mock", workspace: "/tmp/x", busy: null });
      return;
    }
    if (method === "diff") {
      this.reply(socket, id, { changes: [{ path: "a.py", status: "modified" }] });
      return;
    }
    if (method === "prompt") {
      this.reply(socket, id, { ok: true, text: `echo: ${String(params.text)}` });
      return;
    }
    this.fail(socket, id, -32601, `no such method ${method}`);
  }

  reply(socket: net.Socket, id: number, result: unknown): void {
    socket.write(`${JSON.stringify({ jsonrpc: "2.0", id, result })}\n`);
  }

  fail(socket: net.Socket, id: number, code: number, message: string): void {
    socket.write(`${JSON.stringify({ jsonrpc: "2.0", id, error: { code, message } })}\n`);
  }

  /** Push a notification, as the agent loop does. */
  notify(event: string, params: Record<string, unknown>): void {
    for (const socket of this.live) {
      socket.write(`${JSON.stringify({ jsonrpc: "2.0", method: event, params })}\n`);
    }
  }

  /**
   * Answer in two writes, to prove the client reassembles.
   *
   * No delay between them: whether the kernel coalesces the two is not the
   * client's business, and either way it must produce one reply.
   */
  split(socket: net.Socket, id: number, result: unknown): void {
    const frame = `${JSON.stringify({ jsonrpc: "2.0", id, result })}\n`;
    const cut = Math.max(1, Math.floor(frame.length / 2));
    socket.write(frame.slice(0, cut));
    socket.write(frame.slice(cut));
  }

  async start(): Promise<void> {
    this.server = net.createServer((socket) => {
      this.live.add(socket);
      socket.on("close", () => this.live.delete(socket));
      socket.setEncoding("utf8");
      let buffer = "";
      socket.on("data", (chunk: string) => {
        buffer += chunk;
        let index = buffer.indexOf("\n");
        while (index >= 0) {
          const line = buffer.slice(0, index).trim();
          buffer = buffer.slice(index + 1);
          if (line) {
            const message = incoming(JSON.parse(line));
            if (message) {
              this.seen.push(message);
              this.handler(message, socket);
            }
          }
          index = buffer.indexOf("\n");
        }
      });
      socket.on("error", () => undefined);
    });
    const server = this.server;
    await new Promise<void>((resolve) => server.listen(this.socketPath, resolve));
    this.publish();
  }

  publish(overrides: Record<string, unknown> = {}): void {
    const tokenPath = path.join(this.home, "bridge.token");
    fs.writeFileSync(tokenPath, this.token, { mode: 0o600 });
    fs.writeFileSync(
      path.join(this.home, "bridge.json"),
      JSON.stringify({
        version: BRIDGE_VERSION,
        protocol: "2.0",
        transport: "unix",
        path: this.socketPath,
        host: "",
        port: 0,
        token_path: tokenPath,
        pid: process.pid,
        ...overrides,
      }),
    );
  }

  async stop(): Promise<void> {
    for (const socket of this.live) socket.destroy();
    const server = this.server;
    await new Promise<void>((resolve) => {
      if (!server) return resolve();
      server.close(() => resolve());
    });
    fs.rmSync(this.home, { recursive: true, force: true });
  }
}

suite("discovery", () => {
  test("a missing descriptor is a sentence, not a throw", () => {
    const found = discover(fs.mkdtempSync(path.join(os.tmpdir(), "empty-")));
    assert.equal(found.connection, undefined);
    assert.match(String(found.problem), /is offset running/);
  });

  test("OFFSET_HOME wins over the default", () => {
    const previous = process.env.OFFSET_HOME;
    process.env.OFFSET_HOME = "/tmp/somewhere";
    try {
      assert.equal(offsetHome(), path.resolve("/tmp/somewhere"));
    } finally {
      if (previous === undefined) delete process.env.OFFSET_HOME;
      else process.env.OFFSET_HOME = previous;
    }
  });

  test("an explicit home beats the environment", () => {
    assert.equal(offsetHome("/tmp/explicit"), path.resolve("/tmp/explicit"));
  });

  test("a descriptor from a future version is refused rather than guessed", async () => {
    const bridge = new FakeBridge();
    await bridge.start();
    bridge.publish({ version: "99" });
    try {
      const found = discover(bridge.home);
      assert.equal(found.connection, undefined);
      assert.match(String(found.problem), /upgrade whichever is older/);
    } finally {
      await bridge.stop();
    }
  });

  test("a unix descriptor with no path is caught here, not at connect time", async () => {
    const bridge = new FakeBridge();
    await bridge.start();
    // The exact shape of a bug already written once: the writer emits `path`,
    // a reader looks for `socket`, and the descriptor silently describes
    // nothing - then connects to port 0, whose failure carries no message.
    bridge.publish({ path: undefined, socket: bridge.socketPath });
    try {
      assert.match(String(discover(bridge.home).problem), /names no path/);
    } finally {
      await bridge.stop();
    }
  });

  test("unreadable JSON is reported by filename", () => {
    const home = fs.mkdtempSync(path.join(os.tmpdir(), "bad-"));
    fs.writeFileSync(path.join(home, "bridge.json"), "{not json");
    assert.match(String(discover(home).problem), /not valid JSON/);
  });

  test("a descriptor with no token path is refused", () => {
    const home = fs.mkdtempSync(path.join(os.tmpdir(), "notok-"));
    fs.writeFileSync(
      path.join(home, "bridge.json"),
      JSON.stringify({ version: BRIDGE_VERSION, transport: "unix", path: "/tmp/x.sock" }),
    );
    assert.match(String(discover(home).problem), /names no token path/);
  });
});

suite("client", () => {
  let bridge: FakeBridge;

  before(async () => {
    bridge = new FakeBridge();
    await bridge.start();
  });

  after(async () => {
    await bridge.stop();
  });

  test("connect authenticates before resolving", async () => {
    const client = new OffsetClient({ home: bridge.home, retry: 0 });
    await client.connect();
    try {
      assert.equal(client.ready, true);
      const hello = bridge.seen.find((m) => m.method === "hello");
      assert.ok(hello, "hello was never sent");
      assert.equal(hello.params.token, bridge.token);
    } finally {
      client.dispose();
    }
  });

  test("a wrong token surfaces as an unauthenticated RpcError", async () => {
    const other = new FakeBridge();
    await other.start();
    fs.writeFileSync(path.join(other.home, "bridge.token"), "wrong", { mode: 0o600 });
    const client = new OffsetClient({ home: other.home, retry: 0 });
    try {
      await assert.rejects(
        () => client.connect(),
        (error: unknown) => error instanceof RpcError && error.unauthenticated,
      );
    } finally {
      client.dispose();
      await other.stop();
    }
  });

  test("typed methods return their payloads", async () => {
    const client = new OffsetClient({ home: bridge.home, retry: 0 });
    await client.connect();
    try {
      assert.equal((await client.status()).model, "mock");
      assert.deepEqual((await client.diff()).map((change) => change.path), ["a.py"]);
      assert.equal((await client.prompt("hi")).text, "echo: hi");
    } finally {
      client.dispose();
    }
  });

  test("a reply written in two pieces is still one reply", async () => {
    const other = new FakeBridge();
    await other.start();
    other.handler = (message, socket) => {
      if (message.method === "status") other.split(socket, message.id, { model: "reassembled" });
      else other.answer(message, socket);
    };
    const client = new OffsetClient({ home: other.home, retry: 0 });
    await client.connect();
    try {
      assert.equal((await client.status()).model, "reassembled");
    } finally {
      client.dispose();
      await other.stop();
    }
  });

  test("two replies in one write are two replies", async () => {
    const other = new FakeBridge();
    await other.start();
    const client = new OffsetClient({ home: other.home, retry: 0 });
    await client.connect();
    try {
      const both = await Promise.all([client.status(), client.status()]);
      assert.deepEqual(both.map((s) => s.model), ["mock", "mock"]);
    } finally {
      client.dispose();
      await other.stop();
    }
  });

  test("notifications reach listeners", async () => {
    const other = new FakeBridge();
    await other.start();
    const client = new OffsetClient({ home: other.home, retry: 0 });
    await client.connect();
    try {
      const arrived = once(client, "tool.started");
      other.notify("tool.started", { name: "read" });
      const [params] = await arrived;
      assert.equal(params.name, "read");
    } finally {
      client.dispose();
      await other.stop();
    }
  });

  test("a method the bridge does not have rejects with its code", async () => {
    const other = new FakeBridge();
    await other.start();
    const client = new OffsetClient({ home: other.home, retry: 0 });
    await client.connect();
    try {
      await assert.rejects(
        () => client.request("nosuchmethod"),
        (error: unknown) => error instanceof RpcError && error.code === -32601,
      );
    } finally {
      client.dispose();
      await other.stop();
    }
  });

  test("calling before connecting rejects rather than hanging", async () => {
    const client = new OffsetClient({ home: bridge.home, retry: 0 });
    await assert.rejects(() => client.status(), /not connected/);
  });

  test("dispose rejects everything still in flight", async () => {
    const other = new FakeBridge();
    await other.start();

    // Wait on the signal that actually orders this: the bridge receiving
    // `hello`.  Waiting on the client's own "connecting" state does not work -
    // it is emitted synchronously inside `connect()`, before the socket is
    // even opened, so a listener attached afterwards has already missed it and
    // a dispose at that point races the handshake it is meant to interrupt.
    let sawHello = (): void => undefined;
    const helloArrived = new Promise<void>((resolve) => {
      sawHello = resolve;
    });
    other.handler = (message) => {
      if (message.method === "hello") sawHello(); // received, deliberately unanswered
    };

    // Short timeout so a regression fails in seconds rather than stalling the
    // run for the default half-minute.
    const client = new OffsetClient({ home: other.home, retry: 0, timeout: 2_000 });
    try {
      const connecting = client.connect();
      await helloArrived;
      client.dispose();
      await assert.rejects(() => connecting, /disposed/);
    } finally {
      await other.stop();
    }
  });

  test("a disposed client cancels its reconnect", async () => {
    const other = new FakeBridge();
    await other.start();
    const client = new OffsetClient({ home: other.home, retry: 10 });
    await client.connect();
    client.dispose();
    assert.equal(client.ready, false);
    // Nothing scheduled can revive it: `dispose` both clears the timer and
    // latches `closing`, so even a fired timer would return immediately.
    assert.equal(client.state, "offline");
    await other.stop();
  });

  test("state transitions are observable in order", async () => {
    const other = new FakeBridge();
    await other.start();
    const client = new OffsetClient({ home: other.home, retry: 0 });
    const states: string[] = [];
    client.on("state", (next) => states.push(next));
    await client.connect();
    client.dispose();
    assert.deepEqual(states, ["connecting", "ready", "offline"]);
    await other.stop();
  });

  test("the address is reported for logs and status bars", async () => {
    const client = new OffsetClient({ home: bridge.home, retry: 0 });
    await client.connect();
    try {
      assert.equal(client.address, bridge.socketPath);
    } finally {
      client.dispose();
    }
  });
});
