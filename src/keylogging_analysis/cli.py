"""keylog-metrics: message-level fluency indicators from a keystroke-logging export."""
import argparse
import sys
from pathlib import Path

from .adapters import ADAPTERS, get_adapter
from .config import MetricConfig
from .metrics import compute_message_metrics
from .provenance import build_provenance, provenance_path, write_provenance


def parse_filters(items) -> dict:
    filters = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--filter expects COL=VALUE, got '{item}'")
        col, value = item.split("=", 1)
        filters.setdefault(col, []).append(value)
    return filters


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="keylog-metrics", description=__doc__)
    p.add_argument("adapter", choices=sorted(ADAPTERS), help="input format")
    p.add_argument("input", type=Path, help="export directory or file, depending on the adapter")
    p.add_argument("--out", type=Path, required=True, help="output CSV (provenance JSON is written next to it)")
    p.add_argument("--config", type=Path, help="MetricConfig as JSON (defaults otherwise)")
    p.add_argument("--filter", action="append", default=[], metavar="COL=VALUE",
                   help="keep messages whose COL equals VALUE; repeat for several values")
    args = p.parse_args(argv)

    config = MetricConfig.from_json(args.config) if args.config else MetricConfig()
    res = get_adapter(args.adapter)(args.input, parse_filters(args.filter) or None)
    table, report = compute_message_metrics(res.data, config, adapter_counts=res.counts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    full_argv = ["keylog-metrics", *(argv if argv is not None else sys.argv[1:])]
    prov = build_provenance(adapter=args.adapter, inputs=res.inputs, config=config,
                            report=report, n_rows_out=len(table), argv=full_argv)
    write_provenance(prov, provenance_path(args.out))
    print(f"keylog-metrics: {len(table)} messages, {report.n_events_out} events "
          f"({report.n_nochange_dropped} no-change dropped) -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
