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

```bash
# Install globally via pipx (Python 3.11+ required)
pipx install git+https://github.com/The-Masked-Bear/offset-terminal.git

# Launch
offset
```

*On first startup, sign in with your **Google** or **GitHub** account. No license codes needed — if your account has an active Offset Plus subscription, it unlocks automatically.*

---

## COMMAND MATRIX

| Command | Tier | What It Does |
| :--- | :---: | :--- |
| `offset` | **Lite** | Start interactive terminal coding session |
| `/spec <N> <task>` | **Plus** | **Speculative Branching**: Fork N parallel worktrees, race models, merge winner |
| `/flow <task>` | **Plus** | **Multi-Model Pipeline**: Planner → Implementer → Critic orchestration |
| `/model` | **Lite** | Interactive model picker across 12+ providers |
| `/login` | **Lite** | Manage API credentials (OpenAI, Anthropic, Google, Ollama) |
| `offset login` | **All** | Sign in with Google or GitHub account |
| `offset sync` | **All** | Sync subscription status |

---

## LITE vs PLUS

| Feature | Offset Lite (Free) | Offset Plus (Subscription) |
| :--- | :---: | :---: |
| **Pricing** | Free Forever | Monthly or Yearly (Cancel Anytime) |
| **Bring Your Own API Keys** | ✅ | ✅ |
| **Terminal Interface & REPL** | ✅ | ✅ |
| **12+ Model Support** | ✅ | ✅ |
| **Easter Egg Engine** | ✅ | ✅ |
| **Speculative Branching (`/spec`)** | ❌ | ✅ |
| **Multi-Model Pipeline (`/flow`)** | ❌ | ✅ |
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
