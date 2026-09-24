import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

from keylogging_analysis.clean import CleaningReport
from keylogging_analysis.config import MetricConfig
from keylogging_analysis.provenance import (build_provenance, provenance_path, sha256_file,
                                            write_provenance)


def test_sha256_file(tmp_path):
    p = tmp_path / "x.csv"
    p.write_bytes(b"abc")
    assert sha256_file(p) == hashlib.sha256(b"abc").hexdigest()


def test_provenance_path():
    assert provenance_path(Path("/a/b/metrics.csv")) == Path("/a/b/metrics.provenance.json")


def test_build_and_write_provenance(tmp_path):
    src = tmp_path / "in.csv"
    src.write_text("hello")
    out_csv = tmp_path / "m.csv"
    out_csv.write_text("message_id\n1\n2\n")
    report = CleaningReport(n_messages=2, n_events_in=5, adapter_counts={"states_kept": 5})
    prov = build_provenance(adapter="languagelab_export", inputs=[src],
                            config=MetricConfig(), report=report, n_rows_out=2,
                            output_path=out_csv,
                            argv=["keylog-metrics", "languagelab_export", str(tmp_path)])
    assert prov["engine"] == "keylogging_analysis"
    assert prov["adapter"] == "languagelab_export"
    assert prov["inputs"] == [{"path": str(src), "sha256": sha256_file(src), "bytes": 5}]
    assert prov["config"]["pause_thresholds_ms"] == [200, 2000]
    assert prov["cleaning"]["adapter_counts"] == {"states_kept": 5}
    assert prov["rows_out"] == 2
    assert set(prov["git"]) == {"commit", "dirty", "source"}
    # output.sha256 is computed over the CSV as actually written, after the fact
    assert prov["output"] == {"sha256": sha256_file(out_csv)}
    assert prov["versions"]["pyarrow"] == version("pyarrow")
    out = tmp_path / "m.provenance.json"
    write_provenance(prov, out)
    assert json.loads(out.read_text(encoding="utf-8")) == prov


def test_build_provenance_without_output_path_omits_output_section(tmp_path):
    # output_path is optional: build_provenance can be called before the CSV
    # exists (e.g. in a unit test), and the "output" section is then omitted
    # rather than computed against a nonexistent file.
    prov = build_provenance(adapter="x", inputs=[], config=MetricConfig(),
                            report=CleaningReport(), n_rows_out=0)
    assert "output" not in prov
    assert prov["versions"]["pyarrow"] == version("pyarrow")


def test_git_state_in_checkout():
    # the test suite runs from a git checkout of this repo
    prov = build_provenance(adapter="x", inputs=[], config=MetricConfig(),
                            report=CleaningReport(), n_rows_out=0)
    assert prov["git"]["commit"] is None or len(prov["git"]["commit"]) == 40
    assert prov["git"]["source"] == "checkout"


def test_git_state_direct_url_mode(tmp_path, monkeypatch):
    """Installed from a git URL (uv tool run --from git+...@v0.1.0): no .git
    checkout, but importlib.metadata's direct_url.json records the commit."""
    import keylogging_analysis.provenance as provenance

    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)  # no .git here

    direct_url = {
        "url": "https://example.invalid/keylogging_analysis.git",
        "vcs_info": {"vcs": "git", "commit_id": "a" * 40, "requested_revision": "v0.1.0"},
    }

    class FakeDistribution:
        def read_text(self, filename):
            assert filename == "direct_url.json"
            return json.dumps(direct_url)

    monkeypatch.setattr(provenance, "distribution", lambda name: FakeDistribution())

    prov = build_provenance(adapter="x", inputs=[], config=MetricConfig(),
                            report=CleaningReport(), n_rows_out=0)
    assert prov["git"] == {"commit": "a" * 40, "requested_revision": "v0.1.0",
                           "source": "direct_url", "dirty": None}


def test_git_state_direct_url_mode_missing_requested_revision(tmp_path, monkeypatch):
    import keylogging_analysis.provenance as provenance

    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)

    direct_url = {"url": "https://example.invalid/x.git",
                  "vcs_info": {"vcs": "git", "commit_id": "b" * 40}}

    class FakeDistribution:
        def read_text(self, filename):
            return json.dumps(direct_url)

    monkeypatch.setattr(provenance, "distribution", lambda name: FakeDistribution())

    prov = build_provenance(adapter="x", inputs=[], config=MetricConfig(),
                            report=CleaningReport(), n_rows_out=0)
    assert prov["git"] == {"commit": "b" * 40, "source": "direct_url", "dirty": None}


def test_git_state_missing_direct_url_stays_none_no_exception(tmp_path, monkeypatch):
    import keylogging_analysis.provenance as provenance

    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)  # no .git

    def raise_not_found(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(provenance, "distribution", raise_not_found)

    prov = build_provenance(adapter="x", inputs=[], config=MetricConfig(),
                            report=CleaningReport(), n_rows_out=0)
    assert prov["git"]["commit"] is None
    assert prov["git"]["dirty"] is None


def test_git_state_invalid_direct_url_json_stays_none_no_exception(tmp_path, monkeypatch):
    import keylogging_analysis.provenance as provenance

    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)

    class FakeDistribution:
        def read_text(self, filename):
            return "not valid json {{"

    monkeypatch.setattr(provenance, "distribution", lambda name: FakeDistribution())

    prov = build_provenance(adapter="x", inputs=[], config=MetricConfig(),
                            report=CleaningReport(), n_rows_out=0)
    assert prov["git"]["commit"] is None
    assert prov["git"]["dirty"] is None


@pytest.mark.parametrize("content", ["null", "[]", "42", '"text"', {"vcs_info": []}, UnicodeDecodeError])
def test_git_state_unusable_direct_url_stays_none_no_exception(tmp_path, monkeypatch, content):
    import keylogging_analysis.provenance as provenance

    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)

    class FakeDistribution:
        def read_text(self, filename):
            if content is UnicodeDecodeError:
                raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
            return content if isinstance(content, str) else json.dumps(content)

    monkeypatch.setattr(provenance, "distribution", lambda name: FakeDistribution())

    prov = build_provenance(adapter="x", inputs=[], config=MetricConfig(),
                            report=CleaningReport(), n_rows_out=0)
    assert prov["git"]["commit"] is None


def test_git_state_no_direct_url_file_stays_none_no_exception(tmp_path, monkeypatch):
    import keylogging_analysis.provenance as provenance

    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)

    class FakeDistribution:
        def read_text(self, filename):
            return None  # importlib.metadata returns None when the file is absent

    monkeypatch.setattr(provenance, "distribution", lambda name: FakeDistribution())

    prov = build_provenance(adapter="x", inputs=[], config=MetricConfig(),
                            report=CleaningReport(), n_rows_out=0)
    assert prov["git"]["commit"] is None
    assert prov["git"]["dirty"] is None
