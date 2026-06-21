from datetime import datetime, timedelta, timezone

from hypothesis.detectors import DTDownwardTrendDetector, OddSnapshot


def _snap(odd: float, seconds: int) -> OddSnapshot:
    return OddSnapshot(
        match_id=101,
        market_type="AH",
        line="-0.5",
        side="home",
        odd=odd,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds),
    )


def test_downward_trend_emits_after_three_qualified_drops():
    detector = DTDownwardTrendDetector()

    assert detector.update_odd(_snap(2.00, 0)) is None
    assert detector.update_odd(_snap(1.99, 5)) is None
    assert detector.update_odd(_snap(1.98, 10)) is None

    event = detector.update_odd(_snap(1.97, 15))

    assert event is not None
    assert event.direction_after == "down"
    assert event.consecutive_downs == 3
    assert event.odd_start == 2.00
    assert event.odd_before == 1.98
    assert event.odd_current == 1.97
    assert event.odd_at_reversal == 1.97
    assert event.cumulative_drop_pct < -0.8


def test_downward_trend_resets_on_non_qualified_step():
    detector = DTDownwardTrendDetector()

    assert detector.update_odd(_snap(2.00, 0)) is None
    assert detector.update_odd(_snap(1.99, 5)) is None
    assert detector.update_odd(_snap(1.995, 10)) is None
    assert detector.update_odd(_snap(1.985, 15)) is None
    assert detector.update_odd(_snap(1.975, 20)) is None

    event = detector.update_odd(_snap(1.965, 25))

    assert event is not None
    assert event.consecutive_downs == 3
    assert event.odd_start == 1.995
