import statistics

import pytest

from keylogging_analysis.config import MetricConfig
from keylogging_analysis.metrics.timing import timing_metrics
from helpers import MSG_A, MSG_B, MSG_C, edits_of, isnan, row

# Message A, cleaned (9 events):
#   IKIs = 100, 200, 2200, 100, 100, 100, 300, 400 ; span = 4500 - 1000 = 3500 ms
#   sorted 100,100,100,100,200,300,400,2200 -> mean 437.5, median 150
#   linear quantiles: q25 = 100, q75 = 300 + 0.25*(400-300) = 325 -> iqr 225
#   |x - 150| = 50,50,50,50,50,150,250,2050 -> mad 50
#   pauses >= 200 at events 2 (200, inserts " "), 3 (2200, after " "),
#   7 (300, after " "), 8 (400, "o" after "y") -> 4 pauses, 3100 ms, 3 between words
#   pauses >= 2000: event 3 only
# Message B, cleaned (4 events; the 450 ms no-change state is gone):
#   IKIs = 150, 150, 300 ; span 600 ; one pause >= 200 (300 ms, inserts "!")
# Message C: one event -> no IKI, span 0


def test_timing_message_a():
    r = row(timing_metrics(edits_of(("A", MSG_A)), MetricConfig()), "A")
    assert r["typing_span_ms"] == 3500
    assert r["iki_mean"] == pytest.approx(437.5)
    assert r["iki_median"] == pytest.approx(150)
    assert r["iki_sd"] == pytest.approx(statistics.stdev([100, 200, 2200, 100, 100, 100, 300, 400]))
    assert r["iki_iqr"] == pytest.approx(225)
    assert r["iki_mad"] == pytest.approx(50)
    assert r["n_pauses_200"] == 4
    assert r["pause_time_ms_200"] == pytest.approx(3100)
    assert r["pauses_per_min_200"] == pytest.approx(4 * 60000 / 3500)
    assert r["share_pauses_between_words_200"] == pytest.approx(0.75)
    assert r["n_pauses_2000"] == 1
    assert r["pause_time_ms_2000"] == pytest.approx(2200)
    assert r["pauses_per_min_2000"] == pytest.approx(60000 / 3500)
    assert r["share_pauses_between_words_2000"] == pytest.approx(1.0)


def test_timing_message_b_punctuation_counts_as_between_words():
    r = row(timing_metrics(edits_of(("B", MSG_B)), MetricConfig()), "B")
    assert r["typing_span_ms"] == 600
    assert r["iki_mean"] == pytest.approx(200)
    assert r["n_pauses_200"] == 1
    assert r["pauses_per_min_200"] == pytest.approx(100.0)
    assert r["share_pauses_between_words_200"] == pytest.approx(1.0)
    assert r["n_pauses_2000"] == 0
    assert r["pause_time_ms_2000"] == 0
    assert r["pauses_per_min_2000"] == 0
    assert isnan(r["share_pauses_between_words_2000"])


def test_timing_single_event_is_nan_not_inf():
    r = row(timing_metrics(edits_of(("C", MSG_C)), MetricConfig()), "C")
    # spec §4.1.3: messages with fewer than 2 events keep a row with NaN timing
    # metrics, including typing_span_ms and pause_time_ms_<theta>.
    assert isnan(r["typing_span_ms"])
    assert all(isnan(r[k]) for k in ["iki_mean", "iki_median", "iki_sd", "iki_iqr", "iki_mad"])
    assert r["n_pauses_200"] == 0
    assert isnan(r["pause_time_ms_200"])
    assert isnan(r["pauses_per_min_200"])
    assert isnan(r["share_pauses_between_words_200"])


def test_timing_pause_exactly_at_threshold_counts():
    r = row(timing_metrics(edits_of(("P", [(0, "a"), (200, "ab")])), MetricConfig()), "P")
    assert r["n_pauses_200"] == 1


def test_timing_pause_before_deletion_excluded_from_location_share():
    # pause 500 ms before a backspace, then 500 ms before "c" typed after "a"
    r = row(timing_metrics(edits_of(("D", [(0, "a"), (10, "ab"), (510, "a"), (1010, "ac")])),
                           MetricConfig()), "D")
    assert r["n_pauses_200"] == 2
    assert r["share_pauses_between_words_200"] == pytest.approx(0.0)


def test_timing_custom_thresholds_and_column_order():
    out = timing_metrics(edits_of(("A", MSG_A)), MetricConfig(pause_thresholds_ms=(300,)))
    assert list(out.columns) == ["typing_span_ms", "iki_mean", "iki_median", "iki_sd", "iki_iqr",
                                 "iki_mad", "n_pauses_300", "pause_time_ms_300",
                                 "pauses_per_min_300", "share_pauses_between_words_300"]
    assert out.loc["A", "n_pauses_300"] == 3


@pytest.mark.parametrize("apostrophe", ["'", "’", "´"])
def test_typographic_apostrophes_are_word_separators(apostrophe):
    # "don?t": the 600 ms pause before "t" follows the apostrophe, so it is a
    # between-word pause whichever apostrophe the device produced (' ’ ´).
    msg = [(0, "d"), (100, "do"), (200, "don"), (300, "don" + apostrophe),
           (900, "don" + apostrophe + "t")]
    r = row(timing_metrics(edits_of(("X", msg)), MetricConfig()), "X")
    assert r["n_pauses_200"] == 1
    assert r["share_pauses_between_words_200"] == pytest.approx(1.0)
