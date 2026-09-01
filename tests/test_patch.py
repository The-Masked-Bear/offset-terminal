"""Hash-anchored patching.

A line-number patch is a bet that nothing moved between reading a file and
writing it, and that bet loses whenever a formatter runs, a sibling edit lands,
or the agent's own earlier hunk shifts everything below it. The loss is silent:
the patch applies cleanly to the wrong region.

So an anchor here identifies a region by the hash of its exact bytes and of the
bytes around it. Every test below is really the same question - when the file
is not what the patcher was promised, does it refuse, or does it corrupt
something?
"""

from __future__ import annotations

from offset.tools.patch import (
    Anchor,
    Hunk,
    Patch,
    anchor_for,
    apply,
    digest,
    locate,
)

SOURCE = """def one():
    return 1


def two():
    return 2


def three():
    return 3
"""


def region(text: str, needle: str) -> tuple[int, int]:
    start = text.index(needle)
    return start, start + len(needle)


def anchor(text: str, needle: str, **kw) -> Anchor:
    start, end = region(text, needle)
    return anchor_for(text, start, end, **kw)


def replace(text: str, needle: str, new: str, **kw) -> Patch:
    return Patch(hunks=(Hunk("replace", anchor(text, needle, **kw), new),))


# -- the happy path -----------------------------------------------------------


def test_an_exact_anchor_applies():
    got = apply(SOURCE, replace(SOURCE, "return 1", "return 111"))
    assert got.ok, got.reason
    assert "return 111" in got.text
    assert "return 2" in got.text, "it rewrote more than the anchored region"


def test_the_rest_of_the_file_is_untouched_byte_for_byte():
    got = apply(SOURCE, replace(SOURCE, "return 1", "return 111"))
    assert got.text.replace("return 111", "return 1") == SOURCE


def test_insert_before_puts_it_before():
    a = anchor(SOURCE, "def two():")
    got = apply(SOURCE, Patch(hunks=(Hunk("insert_before", a, "# counts\n"),)))
    assert got.ok, got.reason
    assert got.text.index("# counts") < got.text.index("def two():")


def test_insert_after_puts_it_after():
    a = anchor(SOURCE, "return 1")
    got = apply(SOURCE, Patch(hunks=(Hunk("insert_after", a, "  # one"),)))
    assert got.ok, got.reason
    assert got.text.index("# one") > got.text.index("return 1")


def test_delete_removes_only_the_region():
    a = anchor(SOURCE, "    return 2\n")
    got = apply(SOURCE, Patch(hunks=(Hunk("delete", a, ""),)))
    assert got.ok, got.reason
    assert "return 2" not in got.text
    assert "return 1" in got.text and "return 3" in got.text


# -- refusal ------------------------------------------------------------------


def test_a_changed_region_is_refused_not_applied_anyway():
    """The whole point of the file. A best-effort apply here corrupts code."""
    patch = replace(SOURCE, "return 1", "return 111")
    moved = SOURCE.replace("return 1", "return 42")
    got = apply(moved, patch)
    assert not got.ok
    assert got.text == moved, "a refused patch must not have edited anything"


def test_the_refusal_says_what_changed():
    """"It did not apply" sends somebody hunting; naming the region does not."""
    patch = replace(SOURCE, "return 1", "return 111")
    got = apply(SOURCE.replace("return 1", "return 42"), patch)
    assert not got.ok
    assert got.reason
    assert "return 1" in got.reason or "anchor" in got.reason.lower()


def test_a_file_that_changed_elsewhere_still_applies():
    """Strictness has to stop somewhere useful: if the anchored bytes and the
    bytes around them are identical, an edit far away is not this patch's
    business.

    "Far away" is literal - the anchor carries CONTEXT characters either side,
    so the padding below is what makes the second edit genuinely outside it.
    """
    padding = "\n".join(f"# filler line {n}" for n in range(40))
    big = SOURCE + "\n" + padding + "\n\ndef four():\n    return 4\n"
    patch = replace(big, "return 1", "return 111")
    elsewhere = big.replace("return 4", "return 444")
    got = apply(elsewhere, patch)
    assert got.ok, got.reason
    assert "return 111" in got.text
    assert "return 444" in got.text


def test_a_file_hash_mismatch_is_noted_but_not_fatal():
    patch = Patch(
        hunks=(Hunk("replace", anchor(SOURCE, "return 1"), "return 111"),),
        file_hash=digest("something else entirely"),
    )
    got = apply(SOURCE, patch)
    assert got.ok, got.reason
    assert got.notes, "a whole-file change should be reported even when it is safe"


# -- ambiguity ----------------------------------------------------------------


TWINS = """def a():
    return 0


def b():
    return 0
"""


def test_identical_regions_are_told_apart_by_their_surroundings():
    """Two `return 0` lines. A context-matching patcher picks one at random;
    this must pick the one it was given."""
    second = TWINS.rindex("    return 0")
    a = anchor_for(TWINS, second, second + len("    return 0"))
    got = apply(TWINS, Patch(hunks=(Hunk("replace", a, "    return 99"),)))
    assert got.ok, got.reason
    assert got.text.index("return 99") > got.text.index("def b():")
    assert got.text.count("return 0") == 1


def test_locate_finds_exactly_one_site_for_a_context_anchor():
    second = TWINS.rindex("    return 0")
    a = anchor_for(TWINS, second, second + len("    return 0"))
    assert len(locate(TWINS, a)) == 1


def test_locate_without_context_sees_both_twins():
    """Proves the disambiguation above comes from the context hash and is not
    an accident of search order."""
    second = TWINS.rindex("    return 0")
    a = anchor_for(TWINS, second, second + len("    return 0"))
    assert len(locate(TWINS, a, context=False)) == 2


# -- atomicity ----------------------------------------------------------------


def test_several_hunks_all_apply():
    patch = Patch(hunks=(
        Hunk("replace", anchor(SOURCE, "return 1"), "return 111"),
        Hunk("replace", anchor(SOURCE, "return 3"), "return 333"),
    ))
    got = apply(SOURCE, patch)
    assert got.ok, got.reason
    assert "return 111" in got.text and "return 333" in got.text
    assert got.applied == 2


def test_one_bad_hunk_rejects_the_whole_patch():
    """Half-applied is the worst outcome: the file is in a state neither the
    model nor the user asked for, and nothing recorded which half landed."""
    stale = anchor(SOURCE.replace("return 3", "return 42"), "return 42")
    patch = Patch(hunks=(
        Hunk("replace", anchor(SOURCE, "return 1"), "return 111"),
        Hunk("replace", stale, "nope"),
    ))
    got = apply(SOURCE, patch)
    assert not got.ok
    assert got.text == SOURCE, "the first hunk was applied despite the second failing"
    assert got.applied == 0


def test_hunks_do_not_disturb_each_others_anchors():
    """Every anchor is minted against the original text, so applying the first
    must not invalidate the second by shifting it."""
    patch = Patch(hunks=(
        Hunk("insert_before", anchor(SOURCE, "def two():"), "# a\n"),
        Hunk("insert_before", anchor(SOURCE, "def three():"), "# b\n"),
    ))
    got = apply(SOURCE, patch)
    assert got.ok, got.reason
    assert got.text.index("# a") < got.text.index("def two():")
    assert got.text.index("# b") < got.text.index("def three():")


# -- normalisation ------------------------------------------------------------


def test_the_default_is_byte_exact():
    """Whitespace is meaningful in Python, so a reindent is a real change and
    the strict default is the safe one."""
    patch = replace(SOURCE, "    return 1", "    return 111")
    reindented = SOURCE.replace("    return 1", "        return 1")
    assert not apply(reindented, patch).ok


def test_normalise_forgives_reindentation_when_asked():
    patch = replace(SOURCE, "    return 1", "    return 111", normalise=True)
    reindented = SOURCE.replace("    return 1", "        return 1")
    got = apply(reindented, patch)
    assert got.ok, got.reason
    assert "return 111" in got.text


# -- the tool -----------------------------------------------------------------


def test_the_tool_is_registered_and_declared_a_writer():
    from offset.tools.base import Danger
    from offset.tools.patch import patch_tools

    tools = {t.name: t for t in patch_tools()}
    assert "patch" in tools
    assert tools["patch"].danger >= Danger.WRITE, "a file writer must not be SAFE"


def test_an_empty_patch_is_refused_rather_than_silently_doing_nothing():
    got = apply(SOURCE, Patch(hunks=()))
    assert not got.ok
    assert got.text == SOURCE
