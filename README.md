# OFFSET

<div align="center">

```
   ▄▄▄     ▄▄▄   
  ████▄▄▄▄▄████  
 ██████████████   OFFSET // THE TERMINAL CODING AGENT
 ██ ██████ ██    "WE WRITE CODE. YOU TAKE CREDIT."
 █████ ██ █████  
 ██████████████  
  ████████████   
```

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-yellow.svg?style=for-the-badge&colorA=111111&colorB=FFDE59)](https://www.gnu.org/licenses/agpl-3.0)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-cyan.svg?style=for-the-badge&colorA=111111&colorB=8CFFFB)](https://www.python.org/downloads/)
[![Release](https://img.shields.io/badge/release-v0.1.0-pink.svg?style=for-the-badge&colorA=111111&colorB=FF90E8)](https://github.com/The-Masked-Bear/offset-terminal/releases)
[![UI: Neubrutalist](https://img.shields.io/badge/UI-Neubrutalist-mint.svg?style=for-the-badge&colorA=111111&colorB=B2FF9E)](https://the-masked-bear.github.io/offset-terminal/)

<br>

<img src="assets/hero-banner.png" alt="Offset Landing Page & REPL" width="880" style="border: 3px solid #111; box-shadow: 8px 8px 0px #111;">

<br><br>

**[🌐 LIVE WEBSITE & INTERACTIVE PLAYGROUND](https://the-masked-bear.github.io/offset-terminal/)** • **[💎 GET OFFSET PLUS](https://debarghya47.gumroad.com/l/qzqnxk)**

</div>

---

## ⚡ WHAT IS OFFSET?

**Offset** is a from-scratch terminal coding agent built for developers who care about speed, taste, and parallel problem solving. Built with a unified neubrutalist design DNA (sharp borders, zero blur, high-contrast monospace tokens), its core superpower is **Speculative Branching**:

Rather than betting your codebase on a single model's hallucination, Offset can fork your repository into $N$ isolated git worktrees, dispatch competing models (Claude Sonnet, GPT-4.1, Gemini 3 Pro) in parallel, execute local test suites, and merge the winning implementation automatically.

<div align="center">
<img src="assets/terminal-tui.png" alt="Offset Terminal TUI" width="880" style="border: 3px solid #111; box-shadow: 8px 8px 0px #111;">
</div>

---

## 🚀 INSTALLATION

Offset is distributed via `pipx` for isolated, global terminal execution.

### Linux / macOS / WSL
```bash
pipx install git+https://github.com/The-Masked-Bear/offset-terminal.git
```

### Windows (PowerShell)
```powershell
pipx install git+https://github.com/The-Masked-Bear/offset-terminal.git
```

### Launch
```bash
offset
```
*On first startup, authenticate with your GitHub or Google account to activate **Offset Lite**.*

---

## 🎮 COMMAND MATRIX

| Command | Tier | Description |
| :--- | :---: | :--- |
| `offset` | **Lite** | Start interactive terminal coding session. |
| `/login` | **Lite** | Manage API credentials (OpenAI, Anthropic, Google, Ollama, OpenCode). |
| `/model` | **Lite** | Interactive dropdown model picker across 12+ providers. |
| `/spec <N> <task>` | **Plus** | **Speculative Branching**: Fork $N$ parallel worktrees, run tests, merge winner. |
| `/flow <task>` | **Plus** | **Multi-Model Orchestration**: Planner $\rightarrow$ Implementer $\rightarrow$ Critic pipeline. |
| `offset upgrade <key>` | **All** | Upgrade from Lite to Plus using your Gumroad license key. |
| `offset demo` | **Lite** | Render the 24fps animated neubrutalist design system. |

---

## 💎 TIERS: LITE vs PLUS

| Feature | Offset Lite (Free) | Offset Plus (Premium) |
| :--- | :---: | :---: |
| **License** | Open Source (AGPL-3.0) | Commercial License |
| **API Keys** | Bring Your Own (BYOK) | BYOK + Cloud Key Pool |
| **Neubrutalist Terminal UI** | ✅ | ✅ |
| **Interactive REPL & Subagents** | ✅ | ✅ |
| **Built-in Easter Egg Engine** | ✅ | ✅ |
| **Speculative Branching (`/spec`)** | ❌ | ✅ |
| **Multi-Model Orchestration (`/flow`)** | ❌ | ✅ |
| **Auto-Worktree Diff & Merge** | ❌ | ✅ |
| **Priority Updates & Releases** | ❌ | ✅ |

👉 **[Upgrade to Offset Plus on Gumroad](https://debarghya47.gumroad.com/l/qzqnxk)**

---

## 🥚 EASTER EGGS

Offset ships with a comprehensive built-in easter egg engine. Try typing these into the REPL or on the [live website](https://the-masked-bear.github.io/offset-terminal/):

- `bear` — Sentience inquiry (escalates across 4 attempts).
- `sudo` — Privilege escalation check (*"This incident will be reported to The-Masked-Bear."*).
- `matrix` — Wake up, operator. Full terminal decoding sequence.
- `neofetch` — ASCII system diagnostic readout.
- `hunter2` — Password honeypot.
- `coffee` — HTCPCP RFC 2324 teapot protocol.
- `↑ ↑ ↓ ↓ ← → ← → B A` — The classic Konami god mode code.

---

## 📜 LICENSE

This project is open source and licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.  
`Copyright (C) 2026 The-Masked-Bear`

See the [LICENSE](LICENSE) file for complete details.

---

## 👤 AUTHOR & CREDITS

- Created by **[The Masked Bear](https://github.com/The-Masked-Bear)**.

*"This incident will be reported to The-Masked-Bear."*
