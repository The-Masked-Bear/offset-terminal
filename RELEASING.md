# Releasing offset

## Cutting a release

```bash
# 1. bump both, they must agree or the workflow refuses the tag
vim pyproject.toml offset/__init__.py

# 2. commit, tag, push
git commit -am "release: 0.6.1"
git tag -a v0.6.1 -m "offset 0.6.1"
git push origin main --tags
```

That is the whole ritual. The tag triggers `.github/workflows/release.yml`, which:

1. **Refuses a tag that disagrees** with `pyproject.toml` or `offset/__init__.py`.
   A mislabelled wheel cannot be corrected afterwards — PyPI will not let a
   version number be reused.
2. Runs the full suite on **3.11, 3.12 and 3.13**.
3. Builds the sdist and wheel, runs `twine check --strict` (which catches a
   README that renders badly on PyPI), and **installs the built wheel into a
   fresh venv** to prove the command runs and detects its own install method.
4. Publishes to PyPI and attaches the artefacts to the GitHub release.

Steps 3 and 4 are independent: if PyPI publishing fails, the GitHub release
still gets its artefacts.

## One-time setup: PyPI Trusted Publishing

**This has not been done yet.** Until it is, the `publish-pypi` job fails with
`invalid-publisher` and everything else in the release still succeeds.

Trusted Publishing means PyPI verifies the workflow's OIDC identity directly,
so there is no API token to store in GitHub, leak, or rotate.

Because `offset-terminal` does not exist on PyPI yet, register it as a
**pending publisher** — that is the flow for a project whose first upload will
come from CI:

1. Sign in at <https://pypi.org> (create an account if needed, and turn on 2FA;
   PyPI requires it for publishing).
2. Go to <https://pypi.org/manage/account/publishing/>.
3. Under **Add a new pending publisher**, fill in exactly:

   | Field | Value |
   | :--- | :--- |
   | PyPI Project Name | `offset-terminal` |
   | Owner | `The-Masked-Bear` |
   | Repository name | `offset-terminal` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   The environment name matters: the workflow declares `environment: name: pypi`,
   and PyPI checks that claim. Leaving it blank here will not match.

4. Push a tag. The first successful run creates the project and uploads to it.

### Optional: protect the environment

In the GitHub repository under **Settings → Environments → pypi** you can add a
required reviewer, so a release waits for a human before it uploads. Worth it
once other people can push tags.

### If you would rather use an API token

Trusted Publishing is better, but a token works. Create one scoped to the
project at <https://pypi.org/manage/account/token/>, add it as the repository
secret `PYPI_API_TOKEN`, and give the publish step:

```yaml
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

Then the `id-token: write` permission and the `environment:` block are no
longer needed.

## Testing a release without publishing

```bash
gh workflow run Release          # dry run: builds and checks, publishes nothing
```

`workflow_dispatch` does not match `startsWith(github.ref, 'refs/tags/v')`, so
both publish jobs are skipped.

## The distribution name

The package is **`offset-terminal`** on PyPI, because `offset` is taken there by
an unrelated project. The command and the import package are both still
`offset`.

`offset/core/update.py` knows this:

- `PACKAGE` is the current name, used for the PyPI URL and version lookup.
- `LEGACY_PACKAGE` is `offset`, checked as a fallback so anyone who installed
  from git before the rename is still recognised rather than looking like a
  fresh checkout.
- `detect_install()` builds its upgrade command from whichever name the
  environment actually answers to — asking pip to upgrade a name it has never
  heard of fails with nothing useful.

If the name ever changes again, those three points are what to update, and
`tests/test_update.py` pins all of them.

## After a release

Auto-update does the rest: an installed offset checks in the background, and
installs the new version on the next launch before the shell opens. Users who
installed from the git URL get it too, since `pipx upgrade` re-pulls the branch.
