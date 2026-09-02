"""Security findings, and the false positives that would get them ignored.

Every rule gets two tests: one proving it fires on real vulnerable code, one
proving it stays silent on a safe near-miss. The second is not padding - it is
the whole value of the module. A scanner that reports the safe form of a
pattern gets switched off within a week, and a switched-off scanner is worse
than none because it also sold false comfort.

The near-misses here are not invented. Each one fired when the scanner was
first run over this repository: a rule's own regex, an `eval` inside a string
literal, an XML namespace, a scheme check naming both schemes, and the
scanner's own explanatory comment.
"""

from __future__ import annotations

import pytest

from offset.core.security import (
    HIGH,
    LOW,
    MEDIUM,
    Finding,
    Report,
    audit,
    parse_reply,
    scan,
    scan_diff,
    scan_text,
    security_criterion,
)


def fires(line: str, path: str = "app.py") -> bool:
    return bool(scan_text(line, path))


def rules_for(line: str, path: str = "app.py") -> set[str]:
    return {f.rule for f in scan_text(line, path)}


# -- credentials -------------------------------------------------------------------


def test_a_hardcoded_password_is_reported():
    assert "hardcoded-password" in rules_for('password = "hunter2secret99"')


def test_a_password_used_as_a_dict_key_is_not_a_credential():
    """`{"password": value}` names a field; it does not store a secret.

    The single noisiest false positive of every credential scanner.
    """
    assert not fires('{"password": value}')
    assert not fires('payload = {"password": user_input}')


def test_a_password_read_from_the_environment_is_the_fix_not_the_bug():
    assert not fires('password = os.environ["PW"]')
    assert not fires('password = os.getenv("PW")')


def test_comparing_a_password_is_not_storing_one():
    assert not fires('if password == "x":')


def test_an_obvious_placeholder_is_not_a_credential():
    assert not fires('password = "your-password"')
    assert not fires('api_key = "xxxxxxxxxxxx"')
    assert not fires('token = "${VAULT_TOKEN}"')


def test_an_aws_key_id_is_reported():
    assert "aws-access-key" in rules_for('k = "AKIAIOSFODNN7EXAMPLE"')


def test_a_private_key_block_is_reported():
    assert "private-key" in rules_for('blob = "-----BEGIN RSA PRIVATE KEY-----"')


def test_a_credential_literal_still_counts_inside_a_string():
    """The regression this guards: exempting string literals to silence code
    rules silenced the key rules completely, since a committed key is only ever
    inside a string."""
    assert fires('SECRET = "AKIAIOSFODNN7EXAMPLE"')


def test_a_credential_still_counts_when_commented_out():
    """Commenting the line out does not remove the secret from history."""
    assert fires('# key = "AKIAIOSFODNN7EXAMPLE"')


def test_a_credential_shaped_literal_in_a_test_file_is_expected():
    """A suite that exercises a credential parser must contain credential-shaped
    strings; reporting them teaches people to ignore the scanner."""
    assert not fires('password = "hunter2secret99"', "tests/test_auth.py")
    assert not fires('password = "hunter2secret99"', "conftest.py")


def test_an_injection_in_a_test_file_is_still_reported():
    """A fixture path excuses a fake credential, not a real vulnerability."""
    assert fires("subprocess.run(cmd, shell=True)", "tests/test_run.py")


# -- execution ----------------------------------------------------------------------


def test_shell_true_is_reported():
    assert "shell-injection" in rules_for("subprocess.run(cmd, shell=True)")


def test_a_fully_literal_command_cannot_be_injected():
    assert not fires('os.system("ls")')


def test_eval_on_a_variable_is_reported():
    assert "eval-exec" in rules_for("eval(user_input)")


def test_literal_eval_is_the_safe_form():
    assert not fires("ast.literal_eval(s)")


def test_eval_on_a_literal_is_not_flagged():
    assert not fires('eval("1 + 1")')


def test_pickle_loads_is_reported():
    assert "insecure-deserialise" in rules_for("pickle.loads(blob)")


def test_yaml_load_without_a_loader_is_reported():
    assert "yaml-load" in rules_for("yaml.load(fh)")


def test_yaml_safe_load_is_not_flagged():
    assert not fires("yaml.safe_load(fh)")
    assert not fires("yaml.load(fh, Loader=yaml.SafeLoader)")


# -- transport and filesystem ---------------------------------------------------------


def test_disabled_tls_verification_is_reported():
    assert "tls-verification-off" in rules_for("requests.get(u, verify=False)")


def test_a_plain_http_url_is_reported():
    assert "plaintext-http" in rules_for('requests.get("http://api.example.com/v1")')


def test_a_scheme_check_naming_both_schemes_is_not_a_request():
    assert not fires('if u.startswith(("http://", "https://")):')


def test_an_xml_namespace_is_an_identifier_not_a_url():
    assert not fires('x = \'<Types xmlns="http://schemas.openxmlformats.org/z">\'')


def test_a_loopback_url_is_not_flagged():
    """An OAuth loopback redirect is plain HTTP by specification."""
    assert not fires('uri = "http://localhost:8080/callback"')
    assert not fires('uri = "http://127.0.0.1:8080/callback"')


def test_a_world_writable_chmod_is_reported():
    assert "world-writable" in rules_for("os.chmod(p, 0o777)")


def test_a_private_chmod_is_not_flagged():
    assert not fires("os.chmod(p, 0o600)")


def test_mktemp_is_reported():
    assert "insecure-temp" in rules_for("p = tempfile.mktemp()")


def test_mkstemp_is_the_safe_form():
    assert not fires("fd, p = tempfile.mkstemp()")


def test_binding_every_interface_is_reported():
    assert "binding-all-interfaces" in rules_for('app.run(host="0.0.0.0")')


def test_binding_loopback_is_not_flagged():
    assert not fires('app.run(host="127.0.0.1")')


# -- where a match does not count -------------------------------------------------------


def test_code_inside_a_string_literal_is_not_code():
    """`"def f(): return eval(x)"` contains no eval.  This fired on a demo
    fixture in this repository."""
    assert not fires('sample = "def f(): return eval(x)"')


def test_a_rules_own_pattern_does_not_report_itself():
    """The scanner reported its own regex the first time it ran over its own
    source.  A pattern that matches a call is not a call."""
    assert not fires('PAT = re.compile(r"pickle\\.loads")')


def test_commented_out_code_is_not_code():
    assert not fires("# we could eval(x) here but do not")


def test_a_hash_inside_a_string_does_not_start_a_comment():
    """Naive comment stripping would blind the scanner to everything after a
    `#` that happens to sit inside a string."""
    assert fires('h = "#not-a-comment"; eval(z)')


def test_a_docstring_example_is_documentation():
    text = '"""Do not write:\n\n    subprocess.run(x, shell=True)\n"""\n'
    assert scan_text(text, "app.py") == []


def test_the_suppression_comment_works():
    assert not fires('password = "hunter2secret99"  # noqa: security')


def test_a_bare_noqa_does_not_silence_a_security_finding():
    """Silencing a lint should not silently silence a vulnerability."""
    assert fires('password = "hunter2secret99"  # noqa')


def test_the_ignore_file_excludes_matching_paths(tmp_path):
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "lib.py").write_text('password = "hunter2secret99"\n')
    (tmp_path / ".offset-security-ignore").write_text("# vendored\nvendor/*\n")
    assert scan(tmp_path).findings == []


def test_without_the_ignore_file_the_same_tree_reports(tmp_path):
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "lib.py").write_text('password = "hunter2secret99"\n')
    assert scan(tmp_path).findings != []


def test_a_vendored_directory_is_skipped_by_default(tmp_path):
    nested = tmp_path / "node_modules" / "x"
    nested.mkdir(parents=True)
    (nested / "bad.js").write_text('const password = "hunter2secret99";\n')
    assert scan(tmp_path).findings == []


def test_a_binary_suffix_is_not_read(tmp_path):
    (tmp_path / "blob.bin").write_bytes(b'password = "hunter2secret99"\n')
    assert scan(tmp_path).scanned == 0


# -- this repository ---------------------------------------------------------------------


def test_offset_itself_is_clean():
    """The strongest available false-positive test: the whole package.

    Any finding here is either a real vulnerability worth fixing or a rule
    worth narrowing - both are things this test should force a decision about
    rather than let drift.
    """
    from pathlib import Path

    report = scan(Path(__file__).resolve().parent.parent / "offset")
    assert report.scanned > 50, "it scanned almost nothing, so it proves nothing"
    assert report.findings == [], [f.line_text() for f in report.findings]


# -- diffs ---------------------------------------------------------------------------------


DIFF = """\
diff --git a/app.py b/app.py
--- a/app.py
+++ b/app.py
@@ -10,3 +10,4 @@
 def handler(request):
+    password = "hunter2secret99"
     return ok
"""


def test_a_diff_reports_only_added_lines():
    report = scan_diff(DIFF)
    assert [f.rule for f in report.findings] == ["hardcoded-password"]


def test_a_diff_finding_carries_the_new_file_line_number():
    """A finding whose line number is wrong is not evidence, it is a wild goose
    chase - so the hunk header has to be honoured."""
    finding = scan_diff(DIFF).findings[0]
    assert finding.path == "app.py"
    assert finding.line == 11


def test_a_diff_ignores_removed_lines():
    removal = DIFF.replace('+    password = "hunter2secret99"',
                           '-    password = "hunter2secret99"')
    assert scan_diff(removal).findings == []


def test_an_empty_diff_is_not_an_error():
    assert scan_diff("").findings == []


# -- the model pass ----------------------------------------------------------------------------


def test_a_model_finding_is_merged_in():
    reply = "high|app.py|3|race-condition|two writers can interleave and corrupt the file"
    report = audit("x = 1\n", path="app.py", ask=lambda _: reply)
    assert [f.rule for f in report.findings] == ["race-condition"]
    assert report.findings[0].source == "model"


def test_prose_from_the_model_yields_no_findings():
    """A model that ignores the format must not become a fabricated finding."""
    report = audit("x = 1\n", path="app.py",
                   ask=lambda _: "I looked carefully and it seems mostly fine.")
    assert report.findings == []


def test_an_empty_model_reply_is_harmless():
    assert audit("x = 1\n", ask=lambda _: "").findings == []


def test_a_model_that_raises_leaves_the_static_findings_intact():
    """A degraded audit is useful; a crashed audit is not."""
    def broken(_prompt: str) -> str:
        raise RuntimeError("the provider fell over")

    report = audit('password = "hunter2secret99"\n', path="app.py", ask=broken)
    assert [f.rule for f in report.findings] == ["hardcoded-password"]


def test_a_model_restating_a_static_finding_is_not_reported_twice():
    report = audit('password = "hunter2secret99"\n', path="app.py",
                   ask=lambda _: "high|app.py|1|hardcoded-password|it is a secret")
    assert len(report.findings) == 1


def test_a_bad_severity_from_the_model_is_dropped():
    assert parse_reply("catastrophic|app.py|1|x|y") == []


def test_a_non_numeric_line_from_the_model_is_dropped():
    assert parse_reply("high|app.py|somewhere|x|y") == []


def test_no_model_means_static_findings_only():
    report = audit('password = "hunter2secret99"\n', path="app.py")
    assert len(report.findings) == 1
    assert report.findings[0].source == "static"


# -- the report ---------------------------------------------------------------------------------


def finding(severity: str = HIGH, line: int = 1, rule: str = "r") -> Finding:
    return Finding(rule=rule, severity=severity, path="a.py", line=line,
                   evidence="x", impact="y")


def test_worst_severity_is_the_worst_present():
    assert Report([finding(LOW), finding(HIGH)]).worst == HIGH
    assert Report([finding(LOW), finding(MEDIUM)]).worst == MEDIUM
    assert Report([]).worst is None


def test_low_findings_alone_are_still_ok():
    """Worth knowing, not worth blocking a branch over."""
    assert Report([finding(LOW)]).ok is True
    assert Report([finding(MEDIUM)]).ok is False


def test_the_report_names_every_finding_and_its_impact():
    lines = "\n".join(Report([finding()], scanned=1).lines())
    assert "a.py:1" in lines
    assert "y" in lines


def test_an_empty_report_says_how_much_it_looked_at():
    assert "3 file(s)" in Report([], scanned=3).lines()[0]


# -- scoring ------------------------------------------------------------------------------------


def test_no_audit_is_excluded_from_scoring_rather_than_scored_zero():
    """Penalising an unaudited branch makes the winner depend on whether the
    audit was reachable, which is not a property of the code."""
    criterion = security_criterion(None, 2.0)
    assert criterion.applies is False


def test_a_clean_audit_scores_full_marks():
    assert security_criterion(Report([], scanned=4), 2.0).score == 1.0


def test_one_high_finding_is_disqualifying():
    """An average would let a dozen plaintext-HTTP notes outweigh a single
    remote-code-execution path."""
    assert security_criterion(Report([finding(HIGH)]), 2.0).score == 0.0


def test_the_reason_names_the_rule_that_disqualified_it():
    criterion = security_criterion(Report([finding(HIGH, rule="eval-exec")]), 2.0)
    assert "eval-exec" in criterion.reason


def test_medium_findings_reduce_the_score_without_zeroing_it():
    score = security_criterion(Report([finding(MEDIUM)]), 2.0).score
    assert 0.0 < score < 1.0


def test_many_low_findings_cost_less_than_one_medium():
    low = security_criterion(Report([finding(LOW) for _ in range(3)]), 1.0).score
    medium = security_criterion(Report([finding(MEDIUM)]), 1.0).score
    assert low > medium


def test_the_score_never_goes_negative():
    report = Report([finding(MEDIUM, line=n) for n in range(20)])
    assert security_criterion(report, 1.0).score == 0.0


# -- the command ----------------------------------------------------------------------------------


class State:
    def __init__(self, workspace):
        self.workspace = workspace


def test_the_command_reports_a_vulnerable_tree(tmp_path):
    (tmp_path / "app.py").write_text('password = "hunter2secret99"\n')
    from offset.core.security import _audit_command

    out = _audit_command(State(tmp_path), [])
    assert any("hardcoded-password" in line for line in out.lines)


def test_the_command_rejects_a_missing_path(tmp_path):
    from offset.core.security import _audit_command

    out = _audit_command(State(tmp_path), ["nope"])
    assert any("no such path" in line for line in out.lines)


def test_the_command_is_registered_lazily():
    """Building at import time would be a cycle: the shell registry imports
    this module."""
    import offset.core.security as module

    first, second = module.COMMANDS, module.COMMANDS
    assert first is second
    assert [c.name for c in first] == ["audit"]


@pytest.mark.parametrize("path", ["app.py", "lib/thing.py", "x.js", "y.yaml"])
def test_a_finding_always_names_a_checkable_location(path):
    findings = scan_text('password = "hunter2secret99"', path)
    assert findings, path
    assert findings[0].path == path
    assert findings[0].line == 1
    assert findings[0].evidence
    assert findings[0].impact
