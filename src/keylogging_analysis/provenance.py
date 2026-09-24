"""What produced an output file: engine version and commit, config, inputs, cleaning counts."""
import datetime as dt
import hashlib
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def provenance_path(out_csv: Path) -> Path:
    out_csv = Path(out_csv)
    return out_csv.with_name(out_csv.stem + ".provenance.json")


def _pkg_version() -> str:
    try:
        return version("keylogging-analysis")
    except PackageNotFoundError:
        from . import __version__
        return __version__


def _git_state() -> dict:
    if not (REPO_ROOT / ".git").exists():
        return {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain",
                                 "--untracked-files=no"],
                                capture_output=True, text=True, check=True).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def build_provenance(*, adapter, inputs, config, report, n_rows_out, argv=None) -> dict:
    return {
        "engine": "keylogging_analysis",
        "version": _pkg_version(),
        "git": _git_state(),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "adapter": adapter,
        "inputs": [{"path": str(p), "sha256": sha256_file(p), "bytes": Path(p).stat().st_size}
                   for p in inputs],
        "config": config.to_dict(),
        "cleaning": report.to_dict(),
        "rows_out": int(n_rows_out),
        "argv": list(argv) if argv is not None else None,
    }


def write_provenance(prov: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(prov, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                          encoding="utf-8")
