#!/bin/sh
# offset installer.
#
#   curl -fsSL https://raw.githubusercontent.com/The-Masked-Bear/offset-terminal/main/install.sh | sh
#
# POSIX sh, not bash: this is the one script that has to run before the user
# has been asked to install anything, and /bin/sh is the only shell that is
# always there.  No arrays, no [[ ]], no local.
#
# What it does, in order of preference:
#   1. uv    - fastest, isolated, and increasingly what people already have
#   2. pipx  - the standard way to install a Python application
#   3. venv  - a private virtualenv plus a shim, which needs nothing but python
#
# It never uses a bare `pip install --user`: on a PEP 668 system that fails
# with a wall of text about externally-managed environments, and on the systems
# where it works it puts a library into the user's global site-packages, which
# is not what an application wants.

set -eu

REPO="The-Masked-Bear/offset-terminal"
PACKAGE="offset-terminal"
SPEC="git+https://github.com/${REPO}.git"
MIN_MINOR=11

# ---------------------------------------------------------------- appearance

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    B=$(printf '\033[1m'); DIM=$(printf '\033[2m'); R=$(printf '\033[0m')
    OK=$(printf '\033[1;32m'); WARN=$(printf '\033[1;33m'); ERR=$(printf '\033[1;31m')
else
    B=''; DIM=''; R=''; OK=''; WARN=''; ERR=''
fi

say()  { printf '%s\n' "$*"; }
step() { printf '%s==>%s %s\n' "$B" "$R" "$*"; }
good() { printf '%s  ok%s %s\n' "$OK" "$R" "$*"; }
warn() { printf '%s  !%s  %s\n' "$WARN" "$R" "$*"; }
die()  { printf '%s  error%s %s\n' "$ERR" "$R" "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# ------------------------------------------------------------------- options

METHOD=""
FROM_PYPI=0
QUIET=0

usage() {
    cat <<EOF
${B}offset installer${R}

  --method uv|pipx|venv   force one installer instead of choosing
  --pypi                  install the published release rather than git main
  --quiet                 less output
  --help                  this

Environment:
  OFFSET_INSTALL_DIR      where to put the shim for the venv method
                          (default: \$XDG_BIN_HOME, ~/.local/bin, or ~/bin)
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --method) METHOD="${2:-}"; shift 2 ;;
        --method=*) METHOD="${1#*=}"; shift ;;
        --pypi) FROM_PYPI=1; shift ;;
        --git) FROM_PYPI=0; shift ;;
        --quiet|-q) QUIET=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) die "unknown option $1 (try --help)" ;;
    esac
done

[ "$FROM_PYPI" -eq 1 ] && SPEC="$PACKAGE"

# -------------------------------------------------------------------- python

# The newest interpreter that is new enough, not merely the first one found:
# a machine with python3.9 as `python3` and python3.13 alongside should use the
# latter rather than refusing.
find_python() {
    best=""
    best_minor=0
    for candidate in python3.14 python3.13 python3.12 python3.11 python3 python; do
        have "$candidate" || continue
        minor=$("$candidate" -c 'import sys; print(sys.version_info[1] if sys.version_info[0]==3 else -1)' 2>/dev/null) || continue
        [ -z "$minor" ] && continue
        [ "$minor" -lt "$MIN_MINOR" ] 2>/dev/null && continue
        if [ "$minor" -gt "$best_minor" ] 2>/dev/null; then
            best="$candidate"; best_minor="$minor"
        fi
    done
    [ -n "$best" ] && command -v "$best"
}

# ------------------------------------------------------------- shim location

shim_dir() {
    if [ -n "${OFFSET_INSTALL_DIR:-}" ]; then printf '%s' "$OFFSET_INSTALL_DIR"; return; fi
    if [ -n "${XDG_BIN_HOME:-}" ]; then printf '%s' "$XDG_BIN_HOME"; return; fi
    if [ -d "$HOME/.local/bin" ]; then printf '%s' "$HOME/.local/bin"; return; fi
    if [ -d "$HOME/bin" ]; then printf '%s' "$HOME/bin"; return; fi
    printf '%s' "$HOME/.local/bin"
}

on_path() {
    case ":${PATH}:" in *":$1:"*) return 0 ;; *) return 1 ;; esac
}

# Name the file the user actually has, rather than guessing ~/.bashrc at them.
# shellcheck disable=SC2088  # these are shown to a human, not used as paths:
# "add this to ~/.zshrc" reads better than the expanded absolute path.
profile_hint() {
    shell_name=$(basename "${SHELL:-sh}")
    case "$shell_name" in
        zsh)  printf '%s' "~/.zshrc" ;;
        fish) printf '%s' "~/.config/fish/config.fish" ;;
        bash) if [ -f "$HOME/.bash_profile" ]; then printf '%s' "~/.bash_profile"
              else printf '%s' "~/.bashrc"; fi ;;
        *)    printf '%s' "~/.profile" ;;
    esac
}

# ------------------------------------------------------------------ installers

install_uv() {
    step "Installing with uv"
    uv tool install --force "$SPEC" || return 1
    return 0
}

install_pipx() {
    step "Installing with pipx"
    pipx install --force "$SPEC" || return 1
    return 0
}

install_venv() {
    py="$1"
    root="${XDG_DATA_HOME:-$HOME/.local/share}/offset"
    bin_dir=$(shim_dir)

    step "Installing into a private virtualenv"
    say "${DIM}  venv: $root${R}"

    "$py" -m venv --clear "$root/venv" 2>/dev/null || {
        warn "python -m venv failed; on Debian or Ubuntu install python3-venv"
        return 1
    }
    "$root/venv/bin/python" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true
    "$root/venv/bin/python" -m pip install --quiet "$SPEC" || return 1

    mkdir -p "$bin_dir"
    # A shim rather than a symlink: a symlink into the venv resolves argv[0] to
    # the real path on some systems and confuses the venv's own sys.prefix
    # detection.  Two lines of sh cannot get that wrong.
    cat > "$bin_dir/offset" <<EOF
#!/bin/sh
exec "$root/venv/bin/offset" "\$@"
EOF
    chmod +x "$bin_dir/offset"
    good "shim: $bin_dir/offset"

    if ! on_path "$bin_dir"; then
        warn "$bin_dir is not on your PATH"
        say  "     add this to $(profile_hint):"
        say  "       export PATH=\"$bin_dir:\$PATH\""
    fi
    return 0
}

# ----------------------------------------------------------------------- main

say ""
say "${B}offset${R} ${DIM}— terminal coding agent${R}"
say ""

PY=$(find_python || true)
if [ -z "$PY" ]; then
    die "offset needs Python 3.${MIN_MINOR} or newer, and none was found.
     Debian/Ubuntu: sudo apt install python3 python3-venv
     macOS:         brew install python@3.13
     Or see https://www.python.org/downloads/"
fi
[ "$QUIET" -eq 1 ] || good "python: $PY ($("$PY" -c 'import platform;print(platform.python_version())'))"

if [ -n "$METHOD" ]; then
    case "$METHOD" in
        uv)   have uv   || die "--method uv was asked for but uv is not installed" ;;
        pipx) have pipx || die "--method pipx was asked for but pipx is not installed" ;;
        venv) : ;;
        *)    die "unknown method '$METHOD' (uv, pipx or venv)" ;;
    esac
else
    if   have uv;   then METHOD=uv
    elif have pipx; then METHOD=pipx
    else                 METHOD=venv
    fi
fi

installed=0
case "$METHOD" in
    uv)   install_uv   && installed=1 ;;
    pipx) install_pipx && installed=1 ;;
    venv) install_venv "$PY" && installed=1 ;;
esac

# A chosen installer that fails should not end the attempt: falling back to the
# venv method means the install still succeeds on a machine where uv or pipx is
# present but broken.
if [ "$installed" -eq 0 ] && [ "$METHOD" != "venv" ]; then
    warn "$METHOD could not install offset; falling back to a private virtualenv"
    install_venv "$PY" && installed=1
fi
[ "$installed" -eq 1 ] || die "installation failed. Please open an issue:
     https://github.com/${REPO}/issues"

# ---------------------------------------------------------------- verify

say ""
if have offset; then
    version=$(offset update --check 2>/dev/null | head -n 1 || printf 'installed')
    good "$version"
    say ""
    say "  ${B}offset${R}              start a session"
    say "  ${B}offset login${R}        sign in with Google or GitHub"
    say "  ${B}offset --continue${R}   resume where you left off"
    say ""
    say "  ${DIM}docs: https://the-masked-bear.github.io/offset-terminal/${R}"
else
    good "installed"
    warn "the 'offset' command is not on your PATH yet"
    say  "     open a new terminal, or add the directory printed above to PATH"
fi
say ""
