# OFFSET

<div align="center">

```
   ▄▄▄     ▄▄▄   
  ████▄▄▄▄▄████  
 ██████████████   OFFSET // TERMINAL CODING AGENT
 ██ ██████ ██    SPECULATIVE BRANCHING
 █████ ██ █████  
 ██████████████  
  ████████████   
```

[![Official Website](https://img.shields.io/badge/OFFICIAL_WEBSITE-VISIT_PORTAL-black?style=for-the-badge&logo=googlechrome&logoColor=white&colorA=111111&colorB=FFDE59)](https://the-masked-bear.github.io/offset-terminal/)
[![Offset Plus Subscription](https://img.shields.io/badge/OFFSET_PLUS-SUBSCRIBE-black?style=for-the-badge&logo=gumroad&logoColor=white&colorA=111111&colorB=FF90E8)](https://debarghya47.gumroad.com/l/qzqnxk)

<br>

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-yellow.svg?style=for-the-badge&colorA=111111&colorB=FFDE59)](https://www.gnu.org/licenses/agpl-3.0)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-cyan.svg?style=for-the-badge&colorA=111111&colorB=8CFFFB)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/offset-terminal?style=for-the-badge&label=PYPI&colorA=111111&colorB=FF90E8)](https://pypi.org/project/offset-terminal/)
[![Tests](https://img.shields.io/badge/tests-1831_passing-mint.svg?style=for-the-badge&colorA=111111&colorB=B2FF9E)](https://github.com/The-Masked-Bear/offset-terminal/actions)

<br><br>

<img src="assets/terminal-tui.png" alt="Offset Terminal TUI" width="880" style="border: 3px solid #111; box-shadow: 8px 8px 0px #111;">

</div>

---

## THE PROBLEM

Every AI coding agent runs one model at a time. When it hallucinates, you wait. When it fails, you retry. When it loops, you start over. You're betting your entire codebase on a single model's best guess.

## THE SOLUTION: SPECULATIVE BRANCHING

**Offset** forks your git repository into **N isolated worktrees**, dispatches competing models (Claude, GPT, Gemini, DeepSeek, Ollama) to each one **simultaneously**, runs your local test suite in every worktree, and **auto-merges the branch that passes**. One command. Zero hallucination loops.

```
offset> /spec 3 fix the auth memory leak

[SPEC] Forking into 3 isolated worktrees...
  Branch A → Claude Sonnet 4
  Branch B → GPT-4.1
  Branch C → Gemini 3 Pro

[TEST] Running test suite in all 3 worktrees...
  Branch A: 14/14 passed ✓
  Branch B: 11/14 FAILED ✗
  Branch C: 14/14 passed ✓

[MERGE] Branch A auto-merged into main. Branches B, C cleaned up.
```

No other coding agent does this.

---

## HOW OFFSET COMPARES

| Feature | Offset | Cursor | GitHub Copilot | Aider | Claude Code |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Speculative Branching** (race N models in parallel git worktrees) | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Multi-Model /flow** (Planner → Implementer → Critic pipeline) | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Terminal multiplayer** (`/collab` — several humans, one session) | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Loopback** (running code calls the agent's own tools) | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Hash-anchored patching** (refuses a moved region, never guesses) | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Live model discovery** (asks each provider, never a hardcoded list) | ✅ | ❌ | ❌ | ❌ | ❌ |
| **12+ models from one agent** (Claude, GPT, Gemini, Antigravity, Ollama) | ✅ | Partial | ❌ | ✅ | ❌ |
| **Local-first, no Electron** | ✅ | ❌ | ❌ | ✅ | ✅ |
| **Runs on ARM64 / Raspberry Pi** | ✅ | ❌ | ❌ | ✅ | ✅ |
| **Open Source (AGPL-3.0)** | ✅ | ❌ | ❌ | ✅ | ❌ |
| **Built-in Easter Egg Engine** | ✅ | ❌ | ❌ | ❌ | ❌ |

---

## INSTALLATION

**One line, and it works out what you have:**

```bash
curl -fsSL https://raw.githubusercontent.com/The-Masked-Bear/offset-terminal/main/install.sh | sh
```

It prefers `uv`, falls back to `pipx`, and failing both builds a private
virtualenv and drops a shim on your `PATH` — so it works on a machine with no
Python tooling beyond the interpreter itself. Requires **Python 3.11+**.

<details>
<summary><b>Prefer to run it yourself?</b></summary>

```bash
# uv — fastest, isolated
uv tool install offset-terminal

# pipx — the standard way to install a Python application
pipx install offset-terminal

# pip, into a virtualenv you control
python3 -m venv ~/.offset-venv
~/.offset-venv/bin/pip install offset-terminal
~/.offset-venv/bin/offset

# the development tip rather than the release
pipx install git+https://github.com/The-Masked-Bear/offset-terminal.git

# from a checkout
git clone https://github.com/The-Masked-Bear/offset-terminal.git
cd offset-terminal && pipx install -e .
```

**Installer options**

| | |
| :--- | :--- |
| `--method uv\|pipx\|venv` | force one installer instead of choosing |
| `--pypi` | install the published release rather than git main |
| `--quiet` | less output |
| `OFFSET_INSTALL_DIR=…` | where the shim goes (venv method) |

```bash
curl -fsSL https://.../install.sh | sh -s -- --method venv --pypi
```

</details>

> The package is published as **`offset-terminal`** because `offset` was already
> taken on PyPI. The command is still `offset`.

```bash
offset          # start a session
```

*On first startup, sign in with your **Google** or **GitHub** account. No licence
codes needed — if your account has an active Offset Plus subscription, it unlocks
automatically.*

**Offset keeps itself current.** It checks for a release in the background, and
installs a waiting one on the next launch before the shell opens — then
re-executes, so you are already running the new version. It reads a cached
answer rather than the network, so an offline start costs nothing. Opt out with
`OFFSET_NO_AUTO_UPDATE=1`, or `{"update": {"auto": false}}` in
`~/.offset/config.json`. A git checkout is never touched.

---

## WHAT'S NEW IN 0.9

Offset stops being a coding terminal and starts being a platform that checks
its own work. Seven features, and the first two change what `/spec` means.

### Branches are audited and benchmarked, not just tested

`/spec` used to rank branches on verification, regressions, diff size, lint
and a reviewer. None of those notice a branch that passes every test and
introduces a shell injection, or one that fixes the bug and doubles a hot
loop. Two criteria now do.

**Security** runs a static audit of every branch's worktree - hardcoded
credentials, `shell=True` on interpolated input, `eval` on a non-literal,
`pickle.loads`, `yaml.load` without a loader, SQL built by concatenation,
`verify=False`. One high-severity finding zeroes the criterion outright rather
than being averaged away, and at weight 3.0 that costs more than diff, lint
and reviewer preference combined. So a clean branch with a sixty-line diff
beats a one-line branch that unpickles untrusted bytes.

Every rule is paired with a near-miss it must stay silent on, because a
scanner that cries wolf gets switched off: `{"password": value}` is a field
name, `password = os.environ["PW"]` is the fix, an `eval` inside a string
literal is documentation, and an XML namespace is not a plaintext request.
`/audit` runs it by hand, `/audit --diff` over just the pending change.

**Performance** is measured as a distribution - twelve runs, a discarded
warmup, min/median/stdev and peak RSS - and it **refuses to name a winner it
cannot distinguish**. If the two sample sets overlap, the verdict is
`indistinguishable` and the criterion is excluded from the ranking rather than
scored. Reporting "3% faster" from twelve samples on a Raspberry Pi is a
fabricated result, and a ranking built on fabricated results moves
confidently in random directions. `/bench --save` records a baseline; `/bench`
compares against it.

Both criteria are *excluded* when they did not run, never scored zero: whether
an audit was reachable is not a property of your code.

### Where the money went

`Usage` events used to fly past and nothing caught them. `/cost` now answers
what a turn cost, by model, by day, by session, and `/cost failures` lists
every turn that broke and why. `/trace` shows the last turn's span tree -
steps, tool calls, durations, which one failed.

**An unknown price is reported as unknown, never as zero.** A model the table
has not heard of yields no cost figure, and a total that omits one says so
with a `+`. A confident zero looks like an answer and nobody re-checks an
answer. `~/.offset/pricing.json` overrides any price.

### GitLab, first class

Issues, merge requests, pipelines and job logs, with the same shape as the
GitHub support: `/issues`, `/mr`, `/mrs`, `/pipeline`. Self-hosted works via
`GITLAB_HOST`; the token is read from `GITLAB_TOKEN` or `glab`'s own config
and is sent as a header, never in argv and never as a query parameter.

### Issue to pull request

`/issue <number>` reads the issue, restates what it thinks you want, plans,
branches, implements, tests, and opens a pull request - checkpointed at every
stage, so `/issue resume` picks up where a crash or a closed laptop left off.
Works against either forge.

### Watch it from your phone

`/monitor start` serves a single self-contained page - no CDN, no framework -
showing the current turn, background jobs, and token spend, with a button to
cancel a job.

It binds **127.0.0.1 by default**; a wider interface has to be asked for by
name and warns you when you do. Every route including the page needs a bearer
token, generated with `secrets.token_urlsafe`, stored 0600, and compared with
`secrets.compare_digest` - not `==`, which short-circuits on the first wrong
byte and leaks the token's prefix to anyone timing the responses. The one
mutating route additionally requires the token in a *header*, which a
cross-origin navigation cannot set. There is no static file route at all, so
no request path ever reaches the filesystem.

### Cloud and remote execution

`/remote add` registers a machine; `/remote run` executes a task on its daemon
over SSH. The heavy work happens on the box with the cores, the transcript
arrives here.

### A security seat in the pipeline

`ROLES` gained `security` and `researcher`, so `/flow` and `/decompose` can
seat an auditor that is a *different model* from the one that wrote the code.
A model reviewing its own work is not a review.

---

## WHAT'S NEW IN 0.8

Nine features.

### Ghost text

Dim completions ahead of the cursor, from slash commands, workspace paths and
your own history. Right arrow accepts it, `ctrl-right` takes one word. The
engine has a **40 ms deadline**: a suggestion that misses it is simply absent,
never a stutter while typing.

### Hash-anchored patching

A line-number patch bets nothing moved between reading a file and writing it,
and loses silently when a formatter runs or a sibling edit lands. An anchor
identifies a region by the hash of its bytes **and the bytes around it**, so
two identical blocks are told apart and a changed region is *refused* rather
than corrupted. Hunks apply all-or-nothing.

### Architect and coder decomposition

`/decompose <goal>` — an architect plans a dependency graph, coders execute it
in waves. Two units naming the same file are **serialised at planning time**,
because concurrent edits to one file lose one of them and nothing reports it.
Cycles are refused and named. A failed unit blocks its dependents, not the
independent subtrees.

### Infinite sessions

Compaction now happens without being asked, at turn start. Measured on a real
session: **60,566 → 8,351 tokens**, 69 entries summarised, all 95 originals
still on disk and reachable in `/tree`.

### Mid-stream auditing

The provider stream is watched as it arrives and a generation that has plainly
gone wrong is halted, with its evidence named — runaway repetition, or a claim
about a file that is not there. A second model can be sampled as a judge, on a
worker, so it never holds up a token.

**Off by default.** The checks are heuristics and a false halt costs a correct
answer the user cannot recover. That tension is the whole design.

### Filesystem snapshots

Zero-cost workspace isolation via btrfs, zfs, APFS clonefile or reflink,
**probed by trying the cheap operation** rather than guessing from the mount
table. Where none works it falls back to a real copy and says so, rather than
reporting a snapshot as free when it cost a gigabyte. Release is idempotent and
refuses, loudly, to delete the workspace.

### The loopback bridge

Code the agent is executing can call the agent's own tools over an
authenticated socket. The **permission system still applies** — a writing tool
with nobody to approve it is refused, not allowed. The token travels by
environment because argv is world-readable in `/proc`, and nesting is capped by
a counter that fails closed when corrupted.

### Terminal multiplayer

`/collab host`, `/collab join <addr>` — several people, one session, over the
same protocol as the editor bridge. **One driver at a time**, and the second
claimant is told who holds it rather than silently queued. Chat and prompts are
separate, so an observer's aside cannot become an instruction. A peer that
stops reading is dropped rather than stalling the room.

### MCP marketplace

`/market search|info|install|remove|list`. Installing **records** a server; it
never executes anything at install time. Anything off the built-in list is
untrusted until you say otherwise, and a server missing its environment
variables is installed but reported as unconfigured, by name.

---

## WHAT'S NEW IN 0.7

### The model list is asked for, not remembered

A hardcoded catalogue goes stale the day a provider ships something. Against a
real key this machine found **38 Google models live and 4 in the table** — so
offset now asks each provider what it has, caches the answer, and merges it
over the curated list.

```
/models                 # what you can run, plus the curated set
/models gpt-5           # search everything, live included
/models --all           # the lot
/models --refresh       # re-ask now, and say where each list came from
```

The refresh is a background thread, so startup waits for nothing; a cold cache
shows exactly what shipped before. A provider with no key is never asked, and a
provider that fails keeps the models it last reported rather than emptying your
picker. `OFFSET_NO_MODEL_FETCH=1` turns it off entirely.

### Sign in to Google Antigravity

Antigravity used to ask for an API key, which was really the plain Gemini
provider wearing an Antigravity label. Signing in now goes where a signed-in
account actually lives — Cloud Code Assist — with the project discovered once
per token and the model list scoped to your account.

```
/login          # pick google-antigravity, sign in with your browser
```

Google does not publish the client id its own binary carries, and every
community client declines to embed it: a credential lifted out of a shipped
binary gets rotated and takes every user with it. So create a Desktop OAuth
client and register the redirect URI `/login` prints. An API key still works
and takes the ordinary Gemini path, which is the right answer on a headless box.

### Headless daemon

The editor bridge only existed while a TUI was open. Now it does not need one.

```bash
offset daemon                          # unix socket, this machine
offset daemon --listen 127.0.0.1:8791  # a port, for a client across an SSH hop
offset daemon --idle 900               # exit after 15 minutes with no client
```

It prints a descriptor naming where to connect, saves the session on the way
out, and stops on SIGINT, SIGTERM or SIGHUP.

### A TypeScript client

The agent is Python and stays Python. Everything on the *other* side of the
socket is better served by TypeScript, so `@offset/client` is a real package
with no runtime dependencies:

```ts
import { OffsetClient } from "@offset/client";

const offset = new OffsetClient({ name: "my-editor" });
await offset.connect();
offset.on("tool.started", ({ name }) => console.log("running", name));
const { text } = await offset.prompt("add a test for the parser");
```

The VS Code extension is its first consumer — it no longer carries its own copy
of the protocol — and CI typechecks the extension against the client, so a type
that drifts from `bridge.py` breaks the build rather than somebody's editor.

## WHAT'S NEW IN 0.5

### Code intelligence, not text search

A real Language Server Protocol client. It follows imports, re-exports and
shadowing, so it finds the callsites `grep` misses and renames without
corrupting a file that happens to share a name.

```
lsp definition  file=auth.py line=42 symbol=verify
lsp references  file=auth.py line=42 symbol=verify
lsp_edit rename file=auth.py line=42 symbol=verify new_name=check
```

Probes your `PATH` for pyright, pylsp, ruff, tsserver, gopls, rust-analyzer,
clangd, jdtls and lua-ls. Nothing installed for a language? It tells you what
to install instead of failing.

### Real debugging

A Debug Adapter Protocol client, so the agent reads actual program state
instead of adding `print` statements and guessing.

```
debug breakpoint file=app.py line=88
debug launch program=app.py
debug_inspect variables
debug_inspect evaluate expression="user.permissions"
```

### A browser that can actually check your UI

Backend code can be verified by running it. A web page cannot — the only
evidence markup works is a browser rendering it.

```
browser open url=http://localhost:3000
browser snapshot          # accessibility tree with [ref=eN] handles
browser click selector=e5
browser console           # the errors you would otherwise never see
```

The default view is the accessibility tree rather than a screenshot: smaller,
and the model can act on the refs it just read.

### Search that understands code

```
search getUserAuth              # finds get_user_authentication
symbols importers pkg/auth.py   # what breaks if I change this
```

BM25 over an incremental SQLite index, blended with symbol definitions, path
affinity and import proximity. Identifier splitting is what makes the camelCase
query match the snake_case definition.

### GitHub without leaving the terminal

```
/pr                    # summarise the branch, open the PR
/review 12             # prints by default
/review 12 post        # the extra word publishes it
/fix-ci                # finds the failing check, excerpts the error
/resolve-comments 12 reply
```

### Tasks that survive a restart

```
/task implement auth
```

`plan → implement → test → (fix → retest)* → report`, written to disk after
every transition. Close the terminal mid-task and `/task resume <id>` picks up
at the boundary. The fix loop is bounded, and the bound is in the record.

### Background agents, session resume, and self-update

```
/jobs                  # agents that survive closing the terminal
offset --continue      # carry on the last session
offset update          # or let it update itself on startup
```

---

## COMMAND MATRIX

| Command | Tier | What It Does |
| :--- | :---: | :--- |
| `offset` | **Lite** | Start an interactive terminal coding session |
| `offset --continue` | **Lite** | Resume the most recent session |
| `offset --resume <id>` | **Lite** | Resume a specific session |
| `offset daemon` | **Lite** | Run headless for an editor or a remote client |
| `offset update` | **Lite** | Check for and install a newer offset |
| `offset login` | **All** | Sign in with Google or GitHub |
| `/spec <N> <task>` | **Plus** | **Speculative Branching**: fork N worktrees, race models, merge the winner |
| `/flow <task>` | **Plus** | **Multi-Model Pipeline**: Planner → Implementer → Critic |
| `/decompose <goal>` | **Lite** | **Architect + coders**: a dependency graph, executed in parallel waves |
| `/task <goal>` | **Plus** | **Persistent task**: plan, implement, test, fix, retest — survives a restart |
| `/collab host\|join\|drive\|say` | **Lite** | **Multiplayer**: share this session with other humans |
| `/pr` `/review` `/fix-ci` `/resolve-comments` | **Plus** | GitHub-native workflow |
| `/market search\|install\|remove` | **Lite** | MCP marketplace |
| `/compact` | **Lite** | Summarise old history — or let it happen by itself |
| `/jobs` `/job` `/cancel` | **Lite** | Background agents that outlive the terminal |
| `/models [query] [--all] [--refresh]` | **Lite** | Every model, live from each provider |
| `/model` | **Lite** | Interactive model picker |
| `/login` | **Lite** | Manage API credentials |
| `/plugins` | **Lite** | Loaded plugins, load errors, and the trust gate |
| `/mcp reload\|connect\|resources` | **Lite** | MCP servers without a restart |
| `/audit [path\|--diff]` | **Lite** | Static security audit — findings with evidence, not rule names |
| `/bench [--save] [cmd]` | **Lite** | Time a command against its baseline, and say when it cannot tell |
| `/cost [models\|today\|failures]` | **Lite** | What it cost, by model, day or session |
| `/trace` | **Lite** | The last turn's span tree — steps, tool calls, what failed |
| `/issues` `/mr` `/mrs` `/pipeline` | **Lite** | GitLab issues, merge requests and pipelines |
| `/issue <n>` `/issue resume` | **Lite** | Issue → branch → tests → pull request, checkpointed |
| `/monitor start\|stop\|status` | **Lite** | Watch this session from a phone, token-gated, loopback by default |
| `/remote add\|run\|remove` | **Lite** | Run a task on another machine's daemon over SSH |

### Tools the model can call

`read` `write` `edit` `patch` `bash` `glob` `grep` `list` `fetch` `web_search`
`task` `todo` `document` `system` `file` `open` `lsp` `lsp_edit` `debug`
`debug_inspect` `browser` `search` `symbols` `github` `gitlab`

---

## LITE vs PLUS

| Feature | Offset Lite (Free) | Offset Plus (Subscription) |
| :--- | :---: | :---: |
| **Pricing** | Free Forever | Monthly (Cancel Anytime) |
| **Bring Your Own API Keys** | ✅ | ✅ |
| **Terminal Interface & REPL** | ✅ | ✅ |
| **12+ Model Support** | ✅ | ✅ |
| **Code intelligence (LSP)** | ✅ | ✅ |
| **Debugging (DAP)** | ✅ | ✅ |
| **Browser agent** | ✅ | ✅ |
| **Code search & symbol graph** | ✅ | ✅ |
| **MCP servers & plugins** | ✅ | ✅ |
| **Background agents (`/jobs`)** | ✅ | ✅ |
| **Session resume** | ✅ | ✅ |
| **Self-update** | ✅ | ✅ |
| **VS Code companion** | ✅ | ✅ |
| **Easter Egg Engine** | ✅ | ✅ |
| **Ghost text completion** | ✅ | ✅ |
| **Hash-anchored patching** | ✅ | ✅ |
| **Auto-compacting context** | ✅ | ✅ |
| **Filesystem snapshots** | ✅ | ✅ |
| **Loopback bridge** | ✅ | ✅ |
| **Headless daemon** | ✅ | ✅ |
| **Multiplayer (`/collab`)** | ✅ | ✅ |
| **MCP marketplace (`/market`)** | ✅ | ✅ |
| **Architect + coders (`/decompose`)** | ✅ | ✅ |
| **Speculative Branching (`/spec`)** | ❌ | ✅ |
| **Multi-Model Pipeline (`/flow`)** | ❌ | ✅ |
| **Persistent tasks (`/task`)** | ❌ | ✅ |
| **GitHub workflow (`/pr`, `/fix-ci`)** | ❌ | ✅ |
| **Auto-Worktree Diff & Merge** | ❌ | ✅ |
| **Cloud API Key Pool** | ❌ | ✅ |

👉 **[Subscribe to Offset Plus on Gumroad](https://debarghya47.gumroad.com/l/qzqnxk)**

> **⚠️ Before subscribing:** Sign in to Offset first (`offset login`) with your **Google** or **GitHub** account. Use that **exact same email** on Gumroad checkout so your account upgrades automatically.

---

## SUPPORTED PROVIDERS

Since 0.7 the model list is **asked for, not remembered** — offset queries each
provider on a background thread and merges the answer over its curated table.
A model released this morning is in `/models` this afternoon, and one you type
that nothing has heard of still works.

| Provider | Models | Auth | Live listing |
| :--- | :--- | :--- | :---: |
| **Anthropic** | whatever your key can reach | API Key or OAuth | ✅ |
| **OpenAI** | whatever your key can reach | API Key | ✅ |
| **Google** | whatever your key can reach | API Key | ✅ |
| **Google Antigravity** | scoped to your account | OAuth account link | ✅ |
| **OpenRouter** | 400+, no key needed to browse | API Key | ✅ |
| **DeepSeek** | whatever your key can reach | API Key | ✅ |
| **Ollama** | whatever you have pulled | none | ✅ |
| **Claude Pro** | Opus, Sonnet | OAuth via claude.ai | curated |
| **ChatGPT** | GPT-4o, o3 | OAuth via chatgpt.com | curated |
| **OpenCode** | Zen and Go models | API Key | curated |

```
/models gpt-5      # search everything, live included
/models --refresh  # re-ask now, and say where each list came from
```

---

## LICENSE

Open source under **GNU Affero General Public License v3.0 (AGPL-3.0)**.  
`Copyright (C) 2026 The-Masked-Bear`

See the [LICENSE](LICENSE) file for details.

---

## AUTHOR

Created by **[The Masked Bear](https://github.com/The-Masked-Bear)**.
