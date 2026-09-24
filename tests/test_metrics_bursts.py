import pytest

from keylogging_analysis.config import MetricConfig
from keylogging_analysis.metrics.bursts import pburst_metrics, rburst_metrics
from helpers import MSG_A, MSG_B, MSG_C, edits_of, isnan, row

# Message A: bursts start at event 0 and at every pause.
#   θ=200: starts 0,2,3,7,8 -> [0,1] [2] [3,4,5,6] [7] [8]; sizes 2,1,4,1,1
#          net chars (ins - del): 2, 1, 0 (t,h in; h,t out), 1, 1
#   θ=2000: starts 0,3 -> [0..2] size 3 net 3 ; [3..8] size 6 net 4-2 = 2
#   R: runs [0..4] typing, [5,6] revision (starts at the end), [7,8] typing (last: not an R-burst)
# Message B (cleaned): I, I am, I sm (replace mid-text), I sm!
#   θ=200: pause before "!" -> [0,1,2] size 3 net 1+3+0 = 4 ; [3] size 1 net 1
#   θ=2000: one burst of 4, net 5
#   R: [0,1] typing, [2] revision (mid-text), [3] typing (last)


def test_pbursts_message_a():
    r = row(pburst_metrics(edits_of(("A", MSG_A)), MetricConfig()), "A")
    assert r["n_bursts_200"] == 5
    assert r["burst_size_max_200"] == 4
    assert r["burst_size_mean_200"] == pytest.approx(1.8)
    assert r["burst_size_median_200"] == pytest.approx(1)
    assert r["burst_chars_mean_200"] == pytest.approx(1.0)
    assert r["n_bursts_2000"] == 2
    assert r["burst_size_max_2000"] == 6
    assert r["burst_size_mean_2000"] == pytest.approx(4.5)
    assert r["burst_size_median_2000"] == pytest.approx(4.5)
    assert r["burst_chars_mean_2000"] == pytest.approx(2.5)


def test_pbursts_message_b_and_c():
    out = pburst_metrics(edits_of(("B", MSG_B), ("C", MSG_C)), MetricConfig())
    b, c = row(out, "B"), row(out, "C")
    assert (b["n_bursts_200"], b["burst_size_max_200"]) == (2, 3)
    assert b["burst_chars_mean_200"] == pytest.approx(2.5)
    assert (b["n_bursts_2000"], b["burst_size_max_2000"]) == (1, 4)
    assert b["burst_chars_mean_2000"] == pytest.approx(5.0)
    assert (c["n_bursts_200"], c["burst_size_max_200"], c["burst_size_median_200"]) == (1, 1, 1)


def test_pbursts_column_order():
    out = pburst_metrics(edits_of(("A", MSG_A)), MetricConfig(pause_thresholds_ms=(200,)))
    assert list(out.columns) == ["n_bursts_200", "burst_size_max_200", "burst_size_mean_200",
                                 "burst_size_median_200", "burst_chars_mean_200"]


def test_rbursts_message_a_and_b():
    out = rburst_metrics(edits_of(("A", MSG_A), ("B", MSG_B)), MetricConfig())
    a, b = row(out, "A"), row(out, "B")
    assert (a["n_rbursts"], a["rburst_size_max"], a["rburst_size_median"]) == (1, 5, 5)
    assert (a["n_revisions"], a["n_revisions_leading_edge"]) == (1, 1)
    assert (b["n_rbursts"], b["rburst_size_max"], b["rburst_size_median"]) == (1, 2, 2)
    assert (b["n_revisions"], b["n_revisions_leading_edge"]) == (1, 0)


def test_rbursts_none_without_revision():
    r = row(rburst_metrics(edits_of(("C", MSG_C)), MetricConfig()), "C")
    assert (r["n_rbursts"], r["n_revisions"], r["n_revisions_leading_edge"]) == (0, 0, 0)
    assert isnan(r["rburst_size_max"]) and isnan(r["rburst_size_median"])


def test_rbursts_two_revisions():
    # ab, abc, ab (rev 1), abd, abde, abd, ab (rev 2, two deletions), abx
    steps = [(0, "a"), (10, "ab"), (20, "abc"), (30, "ab"), (40, "abd"), (50, "abde"),
             (60, "abd"), (70, "ab"), (80, "abx")]
    r = row(rburst_metrics(edits_of(("R", steps)), MetricConfig()), "R")
    # typing runs before a revision: [a, ab, abc] size 3, [abd, abde] size 2
    assert (r["n_rbursts"], r["rburst_size_max"], r["rburst_size_median"]) == (2, 3, 2.5)
    assert (r["n_revisions"], r["n_revisions_leading_edge"]) == (2, 2)
