#!/bin/sh
# Mirror every GitHub repository to GitLab.
#
# A mirror push, not a plain one: `--mirror` sends every branch, tag and note,
# and makes the GitLab side match the GitHub side exactly. That is what "sync"
# has to mean for a backup to be worth having - a copy missing the branch you
# needed is not a copy.
#
# It is also why this is destructive on the GitLab side. `--mirror` deletes
# refs that no longer exist upstream, so a GitLab project that has diverged
# loses that divergence. The script refuses to touch a project that already has
# commits GitHub does not, unless --force says otherwise.
#
# Needs a GitLab personal access token with `api` (to create the projects) and
# `write_repository` (to push). Create one at:
#   https://gitlab.com/-/user_settings/personal_access_tokens
#
# Usage:
#   GITLAB_TOKEN=glpat-xxxx ./tools/sync-to-gitlab.sh [--dry-run] [--force]
#                                                     [--namespace NAME]
#                                                     [--visibility private|public]

set -eu

DRY=0
FORCE=0
NAMESPACE=""
VISIBILITY="private"
HOST="https://gitlab.com"

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)    DRY=1 ;;
    --force)      FORCE=1 ;;
    --namespace)  NAMESPACE="${2:?--namespace needs a value}"; shift ;;
    --visibility) VISIBILITY="${2:?--visibility needs a value}"; shift ;;
    --host)       HOST="${2:?--host needs a value}"; shift ;;
    -h|--help)    sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)            echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

die() { echo "sync-to-gitlab: $*" >&2; exit 1; }

[ -n "${GITLAB_TOKEN:-}" ] || die "GITLAB_TOKEN is not set.
  Create one at ${HOST}/-/user_settings/personal_access_tokens
  Scopes needed: api, write_repository"

command -v gh   >/dev/null || die "gh is not installed"
command -v git  >/dev/null || die "git is not installed"
command -v curl >/dev/null || die "curl is not installed"

api() {
  # $1 method, $2 path, rest: curl args
  method="$1"; path="$2"; shift 2
  curl -sS -X "$method" \
       -H "PRIVATE-TOKEN: ${GITLAB_TOKEN}" \
       -H "Content-Type: application/json" \
       "${HOST}/api/v4${path}" "$@"
}

# -- who are we -----------------------------------------------------------------

me=$(api GET /user | python3 -c 'import json,sys; print(json.load(sys.stdin).get("username",""))' 2>/dev/null || true)
[ -n "$me" ] || die "the token was rejected by ${HOST}. Check it has the 'api' scope."
[ -n "$NAMESPACE" ] || NAMESPACE="$me"
echo "==> gitlab user:   $me"
echo "==> namespace:     $NAMESPACE"
echo "==> visibility:    $VISIBILITY"
[ "$DRY" = 1 ] && echo "==> DRY RUN - nothing will be created or pushed"

namespace_id=""
if [ "$NAMESPACE" != "$me" ]; then
  namespace_id=$(api GET "/namespaces?search=${NAMESPACE}" \
    | python3 -c "
import json,sys
for n in json.load(sys.stdin):
    if n['full_path'] == '${NAMESPACE}':
        print(n['id']); break
" 2>/dev/null || true)
  [ -n "$namespace_id" ] || die "no namespace '${NAMESPACE}' this token can write to"
fi

# -- what to copy -----------------------------------------------------------------

repos=$(gh repo list --limit 200 --json nameWithOwner,isFork \
  | python3 -c "
import json,sys
for r in json.load(sys.stdin):
    if not r['isFork']:            # a fork is somebody else's history
        print(r['nameWithOwner'])
")
[ -n "$repos" ] || die "gh returned no repositories"
echo "==> repositories:  $(echo "$repos" | wc -l | tr -d ' ')"
echo

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT INT TERM

failed=0
for full in $repos; do
  name=${full#*/}
  echo "--- $full"

  existing=$(api GET "/projects/$(printf '%s/%s' "$NAMESPACE" "$name" | sed 's|/|%2F|g')" \
    | python3 -c 'import json,sys
try:
    d = json.load(sys.stdin)
    print(d.get("id","") if "id" in d else "")
except Exception:
    print("")' 2>/dev/null || true)

  if [ -z "$existing" ]; then
    echo "    project does not exist yet"
    if [ "$DRY" = 0 ]; then
      body=$(python3 -c "
import json
b = {'name': '${name}', 'path': '${name}', 'visibility': '${VISIBILITY}'}
nid = '${namespace_id}'
if nid: b['namespace_id'] = int(nid)
print(json.dumps(b))")
      created=$(api POST /projects -d "$body" \
        | python3 -c 'import json,sys
d=json.load(sys.stdin)
print(d.get("id") or ("ERROR: " + json.dumps(d.get("message") or d)))')
      case "$created" in
        ERROR*) echo "    $created"; failed=$((failed+1)); continue ;;
        *)      echo "    created project $created" ;;
      esac
    else
      echo "    would create it"
    fi
  else
    echo "    project exists (id $existing)"
  fi

  # A mirror clone: every ref, nothing checked out.
  git clone --quiet --mirror "https://github.com/${full}.git" "$work/$name.git" 2>/dev/null \
    || { echo "    clone failed"; failed=$((failed+1)); continue; }

  target="${HOST#https://}"
  remote="https://oauth2:${GITLAB_TOKEN}@${target}/${NAMESPACE}/${name}.git"

  if [ -n "$existing" ] && [ "$FORCE" = 0 ]; then
    # Refuse to overwrite work that only exists on GitLab. `--mirror` would
    # delete it silently, and a backup script that eats the thing it is backing
    # up is worse than no script.
    theirs=$(git --git-dir "$work/$name.git" ls-remote "$remote" 2>/dev/null | awk '{print $1}' | sort)
    if [ -n "$theirs" ]; then
      unknown=$(echo "$theirs" | while read -r sha; do
        git --git-dir "$work/$name.git" cat-file -e "$sha" 2>/dev/null || echo "$sha"
      done)
      if [ -n "$unknown" ]; then
        echo "    REFUSED: gitlab has $(echo "$unknown" | wc -l | tr -d ' ') commit(s) github does not."
        echo "             a mirror push would delete them. re-run with --force to overwrite."
        failed=$((failed+1))
        continue
      fi
    fi
  fi

  if [ "$DRY" = 1 ]; then
    echo "    would mirror-push $(git --git-dir "$work/$name.git" for-each-ref | wc -l | tr -d ' ') refs"
  else
    if git --git-dir "$work/$name.git" push --mirror --quiet "$remote" 2>/dev/null; then
      echo "    pushed $(git --git-dir "$work/$name.git" for-each-ref | wc -l | tr -d ' ') refs -> ${HOST}/${NAMESPACE}/${name}"
    else
      echo "    push failed - does the token have write_repository?"
      failed=$((failed+1))
    fi
  fi
  rm -rf "$work/$name.git"
done

echo
if [ "$failed" -gt 0 ]; then
  echo "==> finished with $failed failure(s)"
  exit 1
fi
echo "==> all repositories synced"
