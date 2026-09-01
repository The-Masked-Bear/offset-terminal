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
[![Release](https://img.shields.io/badge/release-v0.1.0-pink.svg?style=for-the-badge&colorA=111111&colorB=FF90E8)](https://github.com/The-Masked-Bear/offset-terminal/releases)

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
| `offset` | **Lite** | Start interactive terminal coding session |
| `offset --continue` | **Lite** | Resume the most recent session |
| `offset --resume <id>` | **Lite** | Resume a specific session |
| `offset update` | **Lite** | Check for and install a newer offset |
| `/spec <N> <task>` | **Plus** | **Speculative Branching**: fork N worktrees, race models, score and merge the winner |
| `/flow <task>` | **Plus** | **Multi-Model Pipeline**: Planner → Implementer → Critic |
| `/task <goal>` | **Plus** | **Persistent task**: plan, implement, test, fix, retest — survives a restart |
| `/pr` `/review` `/fix-ci` `/resolve-comments` | **Plus** | GitHub-native workflow |
| `/jobs` `/job` `/cancel` | **Lite** | Background agents that outlive the terminal |
| `/resume <n>` | **Lite** | Reopen an earlier session |
| `/plugins` | **Lite** | Loaded plugins, load errors, and the trust gate |
| `/mcp reload\|connect\|resources` | **Lite** | MCP servers without a restart |
| `/model` | **Lite** | Interactive model picker across 12+ providers |
| `/login` | **Lite** | Manage API credentials |
| `offset login` | **All** | Sign in with Google or GitHub |
| `offset sync` | **All** | Sync subscription status |

### Tools the model can call

`read` `write` `edit` `bash` `glob` `grep` `list` `fetch` `web_search` `task`
`todo` `document` `system` `file` `open` — plus, new in 0.5:
**`lsp`** **`lsp_edit`** **`debug`** **`debug_inspect`** **`browser`**
**`search`** **`symbols`** **`github`**

---

## LITE vs PLUS

| Feature | Offset Lite (Free) | Offset Plus (Subscription) |
| :--- | :---: | :---: |
| **Pricing** | Free Forever | Monthly or Yearly (Cancel Anytime) |
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

| Provider | Models | Auth |
| :--- | :--- | :--- |
| **Anthropic** | Claude Opus 4, Sonnet 4, Haiku 3.5 | API Key or OAuth |
| **OpenAI** | GPT-4.1, GPT-4o, o3, o4-mini | API Key |
| **Google** | Gemini 3 Pro, 3 Flash | API Key |
| **Google Antigravity** | Gemini 3.1 Pro, 3 Flash, Flash Lite | OAuth Account Link |
| **Claude Pro** | Claude Opus 4, Sonnet 4 | OAuth via claude.ai |
| **ChatGPT** | GPT-4o, o3 | OAuth via chatgpt.com |
| **Ollama** | Any local model | Local (no key needed) |
| **DeepSeek** | DeepSeek V3 | API Key |
| **OpenCode** | Zen models | API Key |

---

## LICENSE

Open source under **GNU Affero General Public License v3.0 (AGPL-3.0)**.  
`Copyright (C) 2026 The-Masked-Bear`

See the [LICENSE](LICENSE) file for details.

---

## AUTHOR

Created by **[The Masked Bear](https://github.com/The-Masked-Bear)**.
