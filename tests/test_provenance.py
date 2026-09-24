import hashlib
import json
from pathlib import Path

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
    report = CleaningReport(n_messages=2, n_events_in=5, adapter_counts={"states_kept": 5})
    prov = build_provenance(adapter="languagelab_export", inputs=[src],
                            config=MetricConfig(), report=report, n_rows_out=2,
                            argv=["keylog-metrics", "languagelab_export", str(tmp_path)])
    assert prov["engine"] == "keylogging_analysis"
    assert prov["adapter"] == "languagelab_export"
    assert prov["inputs"] == [{"path": str(src), "sha256": sha256_file(src), "bytes": 5}]
    assert prov["config"]["pause_thresholds_ms"] == [200, 2000]
    assert prov["cleaning"]["adapter_counts"] == {"states_kept": 5}
    assert prov["rows_out"] == 2
    assert set(prov["git"]) == {"commit", "dirty"}
    out = tmp_path / "m.provenance.json"
    write_provenance(prov, out)
    assert json.loads(out.read_text(encoding="utf-8")) == prov


def test_git_state_in_checkout():
    # the test suite runs from a git checkout of this repo
    prov = build_provenance(adapter="x", inputs=[], config=MetricConfig(),
                            report=CleaningReport(), n_rows_out=0)
    assert prov["git"]["commit"] is None or len(prov["git"]["commit"]) == 40
