"""Tests for Daily V1×V2 parity hardening (P0.1)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ops.daily_v2.parity_hardening import (
    classify_divergent_order,
    diff_order_sets,
    filter_orders_as_of,
    order_set_hash,
)


def test_order_set_hash_stable_and_order_independent():
    a = order_set_hash(["2", "1", "3"])
    b = order_set_hash(["3", "1", "2"])
    assert a == b
    assert a != order_set_hash(["1", "2"])


def test_hash_differs_when_one_order_differs():
    assert order_set_hash(["1", "2"]) != order_set_hash(["1", "2", "3"])


def test_diff_detects_only_in_v1_and_v2():
    d = diff_order_sets({"1", "2"}, {"2", "3"})
    assert d["only_in_v1"] == ["1"]
    assert d["only_in_v2"] == ["3"]
    assert d["order_set_match"] is False
    assert d["only_in_v1_count"] == 1
    assert d["only_in_v2_count"] == 1


def test_diff_match_when_equal():
    d = diff_order_sets({"a", "b"}, {"b", "a"})
    assert d["order_set_match"] is True
    assert d["v1_order_set_hash"] == d["v2_order_set_hash"]


def test_filter_orders_as_of_excludes_post_cutoff():
    cutoff = datetime(2026, 7, 29, 22, 1, 54, tzinfo=timezone.utc)
    orders = {
        "1": {"created_at": "2026-07-29T21:00:00+00:00", "created_at_dt": datetime(2026, 7, 29, 21, tzinfo=timezone.utc)},
        "2": {"created_at": "2026-07-29T22:21:00+00:00", "created_at_dt": datetime(2026, 7, 29, 22, 21, tzinfo=timezone.utc)},
    }
    out = filter_orders_as_of(orders, as_of=cutoff)
    assert set(out) == {"1"}


def test_post_cutoff_classified_expected_scope():
    cutoff = datetime(2026, 7, 29, 22, 1, 54, 606850, tzinfo=timezone.utc)
    d = classify_divergent_order(
        order={
            "order_id": "1938082582",
            "created_at": "2026-07-29T22:21:51+00:00",
            "policy_version": "H3BUP_vNext_20260629",
            "stake": 10.0,
            "exec_side": "back",
            "status": "LIVE_OK",
        },
        in_v1=False,
        in_v2_full=True,
        in_v2_parity=False,
        parity_as_of=cutoff,
    )
    assert d["classification"] == "EXPECTED_SCOPE_DIFFERENCE"
    assert d["after_parity_cutoff"] is True


def test_no_unknown_for_post_cutoff_case():
    cutoff = datetime(2026, 7, 29, 22, 1, 54, tzinfo=timezone.utc)
    d = classify_divergent_order(
        order={"order_id": "x", "created_at": "2026-07-29T22:30:00+00:00", "policy_version": "H3BUP_vNext_20260629", "stake": 10},
        in_v1=False,
        in_v2_full=True,
        in_v2_parity=False,
        parity_as_of=cutoff,
    )
    assert d["classification"] != "UNKNOWN"


def test_duplicate_ids_do_not_change_hash_cardinality():
    # set semantics
    d = diff_order_sets({"1", "1", "2"}, {"1", "2"})
    assert d["order_set_match"] is True


def test_roi_void_denominator_contract_note():
    # documentation contract retained in performance module
    from ops.daily_v2.performance import compute_settlement_and_performance

    out = compute_settlement_and_performance(
        orders={"1": {"stake": 10.0}, "2": {"stake": 10.0}},
        pnl_by_oid={"1": 5.0, "2": 0.0},
        open_oids=set(),
    )
    assert out["principal_metric"] == "roi_resolved"
    assert out["stake_void"] == 10.0
    assert out["roi_resolved"]["denominator"] == 20.0
    assert "void" in " ".join(out["formulas"]["roi_resolved"].lower().split())
