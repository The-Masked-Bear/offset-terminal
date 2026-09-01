"""Live model discovery.

The bug this defends against is the one a user actually hit: a model shipped by
a provider, usable by typing its id, and invisible in every list offset shows.
So the tests are about what ends up in the merged view and what happens when
the network does not cooperate - not about ranking or presentation.

Every test here uses an injected fetcher.  Nothing reaches a network.
"""

from __future__ import annotations

import json
import threading

import pytest

from offset.providers import catalogue as cat
from offset.providers.registry import ModelInfo


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """Isolate the cache file and keep every credential lookup predictable."""
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path))
    monkeypatch.delenv(cat.NO_FETCH_ENV, raising=False)
    return tmp_path


def fetcher(payloads: dict[str, object]):
    """A fetcher answering from a url-substring table, and recording calls."""
    seen: list[str] = []

    def fetch(url: str, headers: dict[str, str]):
        seen.append(url)
        for fragment, body in payloads.items():
            if fragment in url:
                return body
        raise AssertionError(f"unexpected url {url}")

    fetch.seen = seen  # type: ignore[attr-defined]
    return fetch


# -- shape inference ----------------------------------------------------------


def test_an_unknown_model_still_gets_a_defensible_context_window():
    """A model nobody has heard of must not arrive claiming zero context."""
    info = cat.describe("gpt-5.6-luna", "openai")
    assert info.context > 0
    assert info.max_output > 0
    assert info.provider == "openai"


@pytest.mark.parametrize(
    "model_id, expected",
    [
        ("gemini-3-pro", 1_048_576),
        ("claude-opus-4-6", 200_000),
        ("gpt-4o", 128_000),
        ("o3", 200_000),
    ],
)
def test_context_comes_from_the_family_when_the_api_is_silent(model_id, expected):
    assert cat.describe(model_id, "x").context == expected


def test_a_reported_context_beats_the_inferred_one():
    """Where the API says, the API wins: the table is only for silence."""
    told = cat.describe("gemini-3-pro", "google", context=64_000, output=1_024)
    assert told.context == 64_000
    assert told.max_output == 1_024


@pytest.mark.parametrize(
    "model_id",
    ["text-embedding-3-large", "tts-1", "dall-e-3", "whisper-1", "omni-moderation-latest"],
)
def test_models_that_are_not_chat_models_are_refused(model_id):
    """A listing endpoint returns everything the provider hosts.  Offering an
    embedding model as a coding model only wastes the user's time."""
    assert not cat.usable_as_chat(model_id)


def test_a_real_coding_model_is_not_caught_by_that_filter():
    for model_id in ("gpt-5.6-luna", "claude-opus-5", "gemini-3.1-pro", "o4-mini"):
        assert cat.usable_as_chat(model_id), model_id


def test_a_reasoning_model_is_recognised_as_one():
    assert cat.describe("o3", "openai").thinking
    assert cat.describe("gemini-3-pro", "google").thinking
    assert not cat.describe("gpt-4o-mini", "openai").thinking


# -- listing ------------------------------------------------------------------


def test_a_provider_with_no_credential_is_not_asked(monkeypatch):
    """A listing call without a key earns a 401 every launch and teaches
    nobody anything, so it must not be made at all."""
    monkeypatch.setattr(cat, "credential", lambda p: None)
    fetch = fetcher({})
    listing = cat.fetch_provider("openai", fetch=fetch)
    assert not listing.ok
    assert "no credential" in listing.error
    assert fetch.seen == []


def test_openai_style_listing_is_parsed(monkeypatch):
    monkeypatch.setattr(cat, "credential", lambda p: "key")
    fetch = fetcher({"api.openai.com": {"data": [
        {"id": "gpt-5.6-luna"}, {"id": "text-embedding-3-large"},
    ]}})
    listing = cat.fetch_provider("openai", fetch=fetch)
    assert listing.ok
    assert [m.id for m in listing.models] == ["gpt-5.6-luna"]


def test_google_listing_honours_the_reported_capabilities(monkeypatch):
    """Google says which methods each model supports, so this is a fact and
    not a guess: a model that cannot generateContent is not a chat model."""
    monkeypatch.setattr(cat, "credential", lambda p: "key")
    fetch = fetcher({"generativelanguage": {"models": [
        {"name": "models/gemini-9-pro", "supportedGenerationMethods": ["generateContent"],
         "inputTokenLimit": 2_000_000, "outputTokenLimit": 8_000, "displayName": "Gemini 9"},
        {"name": "models/embedding-004", "supportedGenerationMethods": ["embedContent"]},
    ]}})
    listing = cat.fetch_provider("google", fetch=fetch)
    assert [m.id for m in listing.models] == ["gemini-9-pro"]
    assert listing.models[0].context == 2_000_000
    assert listing.models[0].label == "Gemini 9"


def test_a_provider_that_raises_is_reported_not_propagated(monkeypatch):
    monkeypatch.setattr(cat, "credential", lambda p: "key")

    def boom(url, headers):
        raise OSError("network is down")

    listing = cat.fetch_provider("openai", fetch=boom)
    assert not listing.ok
    assert "network is down" in listing.error
    assert listing.models == ()


# -- the cache ----------------------------------------------------------------


def test_a_listing_survives_a_round_trip_through_the_cache():
    cat.store(cat.Listing("openai", (cat.describe("gpt-5.6-luna", "openai"),), fetched=1000.0))
    back = cat.cached("openai")
    assert back is not None
    assert [m.id for m in back.models] == ["gpt-5.6-luna"]
    assert back.fetched == 1000.0


def test_an_unreadable_cache_is_not_fatal(home):
    (home / "models.json").write_text("{not json", encoding="utf-8")
    assert cat.cached("openai") is None
    assert cat.merged()  # falls back to the static table


def test_a_cache_from_a_future_layout_is_ignored(home):
    (home / "models.json").write_text(
        json.dumps({"version": cat.CACHE_VERSION + 99, "providers": {"openai": {}}}),
        encoding="utf-8",
    )
    assert cat.cached("openai") is None


def test_freshness_uses_a_shorter_deadline_after_a_failure():
    """A machine briefly off the network should be retried soon; a provider
    that answered should not be re-asked for hours."""
    ok = cat.Listing("openai", (), fetched=0.0)
    bad = cat.Listing("openai", (), error="boom", fetched=0.0)
    midway = cat.RETRY_TTL + 1
    assert cat.stale(bad, now=midway)
    assert not cat.stale(ok, now=midway)
    assert cat.stale(ok, now=cat.TTL + 1)


def test_nothing_cached_is_always_stale():
    assert cat.stale(None)


def test_a_failed_refresh_does_not_empty_the_picker(monkeypatch):
    """The regression that would hurt most: one flaky launch wiping the list
    of models the user could pick yesterday."""
    monkeypatch.setattr(cat, "credential", lambda p: "key")
    good = fetcher({"api.openai.com": {"data": [{"id": "gpt-5.6-luna"}]}})
    cat.refresh(["openai"], fetch=good, force=True)
    assert len(cat.cached("openai").models) == 1

    def boom(url, headers):
        raise OSError("down")

    cat.refresh(["openai"], fetch=boom, force=True)
    kept = cat.cached("openai")
    assert [m.id for m in kept.models] == ["gpt-5.6-luna"], "a failure erased the cache"
    assert kept.error, "but the failure is still recorded"


def test_a_fresh_cache_is_not_re_fetched(monkeypatch):
    monkeypatch.setattr(cat, "credential", lambda p: "key")
    fetch = fetcher({"api.openai.com": {"data": [{"id": "a"}]}})
    cat.refresh(["openai"], fetch=fetch, force=True)
    before = len(fetch.seen)
    cat.refresh(["openai"], fetch=fetch)  # not forced, and still fresh
    assert len(fetch.seen) == before


def test_the_fetch_can_be_switched_off_entirely(monkeypatch):
    monkeypatch.setenv(cat.NO_FETCH_ENV, "1")
    fetch = fetcher({})
    assert cat.refresh(fetch=fetch) == []
    assert fetch.seen == []


# -- merging ------------------------------------------------------------------


def test_a_live_model_the_table_never_heard_of_is_added(monkeypatch):
    monkeypatch.setattr(cat, "credential", lambda p: "key")
    cat.refresh(["openai"], fetch=fetcher({"api.openai.com": {"data": [{"id": "gpt-5.6-luna"}]}}),
                force=True)
    assert "gpt-5.6-luna" in {m.id for m in cat.merged()}


def test_the_curated_role_hint_survives_a_live_listing(monkeypatch):
    """The whole reason to keep a static table: an API reports an id, not that
    the model is the one you want planning your work."""
    monkeypatch.setattr(cat, "credential", lambda p: "key")
    monkeypatch.setattr(cat, "MODELS", (
        ModelInfo("gpt-4o", "openai", "gpt-4o", 128_000, 16_384, role_hint="implementer"),
    ))
    cat.refresh(["openai"], fetch=fetcher({"api.openai.com": {"data": [{"id": "gpt-4o"}]}}),
                force=True)
    found = {m.id: m for m in cat.merged()}["gpt-4o"]
    assert found.role_hint == "implementer"
    assert found.label == "gpt-4o"


def test_the_static_table_is_the_floor_when_nothing_has_been_fetched():
    from offset.providers.registry import MODELS

    assert {m.id for m in cat.merged()} == {m.id for m in MODELS}


def test_the_merged_view_never_repeats_an_id(monkeypatch):
    monkeypatch.setattr(cat, "credential", lambda p: "key")
    cat.refresh(["openai"], fetch=fetcher({"api.openai.com": {"data": [{"id": "gpt-4o"}]}}),
                force=True)
    ids = [m.id for m in cat.merged()]
    assert len(ids) == len(set(ids))


def test_the_shipped_table_itself_has_no_duplicates():
    """It had two, for `o3` and `o4-mini`, and nothing noticed."""
    from offset.providers.registry import MODELS

    ids = [m.id for m in MODELS]
    assert len(ids) == len(set(ids)), "duplicate ids in the shipped catalogue"


def test_a_background_refresh_writes_where_it_started_not_where_it_lands(tmp_path, monkeypatch):
    """A daemon thread outlives whatever started it.

    Resolving `settings.home()` inside the worker meant a refresh begun against
    one home finished against another - and the other was observably the user's
    real `~/.offset`, because a test reverting `OFFSET_HOME` (or a shell simply
    exiting) is enough to change the answer mid-flight.
    """
    started_in = tmp_path / "started"
    started_in.mkdir()
    later = tmp_path / "later"
    later.mkdir()

    monkeypatch.setattr(cat, "credential", lambda p: "key")
    monkeypatch.setenv("OFFSET_HOME", str(started_in))

    done = threading.Event()
    cat.refresh_async(
        fetch=fetcher({"api.openai.com": {"data": [{"id": "gpt-5.6-luna"}]}}),
        done=done,
    )
    # Move the goalposts while the worker is in flight, exactly as monkeypatch
    # does when a test ends.
    monkeypatch.setenv("OFFSET_HOME", str(later))
    assert done.wait(20), "the refresh thread never finished"

    assert (started_in / "models.json").exists(), "it did not write where it started"
    assert not (later / "models.json").exists(), "it followed OFFSET_HOME after the fact"


def test_an_explicit_home_beats_the_environment(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(cat, "credential", lambda p: "key")
    cat.refresh(
        ["openai"],
        fetch=fetcher({"api.openai.com": {"data": [{"id": "a"}]}}),
        force=True,
        home=elsewhere,
    )
    assert (elsewhere / "models.json").exists()
    assert cat.cached("openai", elsewhere) is not None
    assert cat.cached("openai") is None, "the default home should be untouched"
