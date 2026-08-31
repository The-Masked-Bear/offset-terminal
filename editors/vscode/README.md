# Offset Companion

A VS Code companion for [offset](https://github.com/The-Masked-Bear/offset-terminal), the
terminal coding agent.

Offset keeps its terminal identity. This extension does not reimplement the agent
UI — there is one conversation and it lives in your terminal. It surfaces the
three things an editor is genuinely better at:

- **Status bar** — current model, agent state, and how many background jobs are running.
- **Pending changes** — the agent's uncommitted edits, openable as a real side-by-side diff.
- **Send selection** — hand the agent a selection along with its file and line, which a
  terminal cannot see for itself.

## How it connects

Offset exposes a local IPC bridge: a Unix domain socket speaking newline-delimited
JSON-RPC, described by `~/.offset/bridge.json` with its token in
`~/.offset/bridge.token` (both mode `0600`).

The bridge can apply edits, so it is **authenticated**. The extension performs the
same handshake any client must: read the descriptor, read the token from the path it
names, connect, then `hello` with the token. Every other method is refused until that
succeeds.

If offset is not running there is nothing to connect to, and that is the normal case —
the status bar says so and one timer retries. It will not spin.

## Commands

| Command | What it does |
|---|---|
| `Offset: Connect to running agent` | Connect, or report why it cannot |
| `Offset: Send selection to agent` | Prompt, with the selection and its file:line |
| `Offset: Show pending changes` | Pick a changed file and open a diff |
| `Offset: Cancel current turn` | Stop the agent mid-turn |
| `Offset: Open a session log` | Pick a session and open its JSONL |
| `Offset: Refresh` | Re-read status and pending changes |

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `offset.home` | `""` | Where offset keeps state. Empty means `$OFFSET_HOME`, then `~/.offset` |
| `offset.autoConnect` | `true` | Connect when the window opens |
| `offset.reconnectSeconds` | `5` | Wait before retrying a dropped connection; `0` disables |

## Building

Requires Node and npm. `node_modules` is deliberately not committed.

```bash
cd editors/vscode
npm install
npm run compile      # or: npm run watch
npm run typecheck    # tsc --noEmit
```

To try it: open this folder in VS Code and press <kbd>F5</kbd> to launch an Extension
Development Host. Start `offset` in a terminal so there is a bridge to connect to.

To package:

```bash
npx @vscode/vsce package
```

## Licence

AGPL-3.0-only, as with the rest of offset.
