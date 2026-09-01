# `@offset/client`

TypeScript client for [offset](https://github.com/The-Masked-Bear/offset-terminal),
the terminal coding agent.

offset's agent is Python and stays Python. What belongs in TypeScript is
everything on the *other* side of the socket — VS Code, JetBrains, a Node
script, a web front end — and this is that half. Zero runtime dependencies,
matching the Python side's own policy: Node built-ins only.

## Install

```bash
npm install @offset/client
```

## Use

Start offset with a socket to talk to. Either a normal session:

```bash
offset            # the TUI publishes a bridge while it runs
```

…or headless, which is what an editor or a remote machine wants:

```bash
offset daemon                          # unix socket, this machine only
offset daemon --listen 127.0.0.1:8791  # a port, for a client across an SSH hop
offset daemon --idle 900               # exit after 15 minutes with no client
```

Then:

```ts
import { OffsetClient } from "@offset/client";

const offset = new OffsetClient({ name: "my-editor" });
await offset.connect();

offset.on("tool.started", ({ name }) => console.log("running", name));
offset.on("agent.finished", ({ steps }) => console.log("done in", steps, "steps"));

console.log(await offset.status());

for (const change of await offset.diff()) {
  console.log(change.status, change.path, `+${change.additions} -${change.deletions}`);
}

const { text } = await offset.prompt("add a test for the parser");
console.log(text);

offset.dispose();
```

## How it finds offset

Three steps, in the order the Python side documents them:

1. Resolve the home directory — `$OFFSET_HOME`, else `~/.offset`.
2. Read `bridge.json`, which says whether to use a unix socket or a port.
3. Read the token from the path that file names.

The token is never inlined in the descriptor, because descriptors get pasted
into bug reports. `discover()` does all three and returns a sentence rather
than throwing when offset simply is not running — the ordinary case.

```ts
import { discover, describe } from "@offset/client";

const found = discover();
if (found.connection) console.log("offset at", describe(found.connection));
else console.log(found.problem);
```

## Authentication

The socket can apply edits and run tools, so it is authenticated: `hello` must
succeed before the bridge dispatches anything. `connect()` does that for you
and only resolves once it has been accepted — resolving earlier would hand back
a client whose every call fails.

A wrong or missing token arrives as an `RpcError` with `unauthenticated` set,
so a client can distinguish "not allowed" from "not running" without matching
on message text.

## Exposing a daemon beyond loopback

`--listen 0.0.0.0:8791` puts an agent that can run shell commands on your
network. The token is mandatory on every transport, but a token is not a reason
to do this casually. Prefer an SSH tunnel:

```bash
ssh -N -L 8791:127.0.0.1:8791 you@buildbox   # locally
offset daemon --listen 127.0.0.1:8791        # on the box
```

## API

| Method | Returns |
| --- | --- |
| `connect(home?)` | resolves once authenticated |
| `status()` | model, session, state, jobs, workspace |
| `sessions()` | recent sessions |
| `diff()` | pending changes, with both texts for a side-by-side view |
| `applyEdit(target, text)` | write one file through the agent |
| `cancel()` | interrupt the running turn |
| `prompt(text, selection?)` | run a turn; resolves when it finishes |
| `request(method, params)` | anything the bridge grows before this client does |
| `dispose()` | stop, and stay stopped |

Events: `agent.started`, `agent.finished`, `tool.started`, `tool.finished`,
`edit.applied`, `job.state`, plus `state` and `closed` from the client itself.

## Protocol

Newline-delimited JSON-RPC 2.0. `src/protocol.ts` is the TypeScript half of
`offset/core/bridge.py` and the only place the wire format is written down on
this side, so a disagreement between the two is a diff in one file rather than
a mystery spread across a client.

## Licence

AGPL-3.0-only, as with the rest of offset.
