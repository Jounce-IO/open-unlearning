"""Binding table and pass specs for multi-pass trajectory eval."""

from evals.metrics.trajectory_pass_binding import (
    DISPLAY_METRIC_BINDING,
    HM_AGGREGATE_SUBMETRIC_BINDING,
    canonical_pass_ids_eight,
    canonical_pass_ids_twelve,
    filter_metrics_and_data_for_pass,
    get_pass_spec,
    list_all_pass_specs,
)


def test_display_metric_binding_count():
    assert len(DISPLAY_METRIC_BINDING) == 8


def test_hm_aggregate_submetric_binding_count():
    assert len(HM_AGGREGATE_SUBMETRIC_BINDING) == 9


def test_canonical_pass_ids_eight():
    ids = canonical_pass_ids_eight()
    assert len(ids) == 8
    assert "forget__unguided" in ids
    assert "wf__guided_native" in ids


def test_canonical_pass_ids_twelve():
    assert len(canonical_pass_ids_twelve()) == 12


def test_get_pass_spec_forget_unguided():
    s = get_pass_spec("forget__unguided")
    assert s.evaluation_mode == "unguided"
    assert "rouge" in s.internal_metric_keys
    assert "forget" in s.dataset_access_keys


def test_filter_forget_unguided():
    m = {"rouge": {}, "probability": {}}
    data = {"forget": object(), "retain": object()}
    tc = {"evaluation_mode": "guided_native"}
    fm, fd, ftc = filter_metrics_and_data_for_pass("forget__unguided", m, data, tc)
    assert set(fm.keys()) == {"rouge"}
    assert set(fd.keys()) == {"forget"}
    assert ftc["evaluation_mode"] == "unguided"


def test_filter_retain_unguided():
    m = {"rouge": {}, "truth_ratio": {}, "privleak": {}, "probability": {}}
    data = {"retain": object(), "forget": object()}
    tc = {"evaluation_mode": "guided_native"}
    fm, fd, ftc = filter_metrics_and_data_for_pass("retain__unguided", m, data, tc)
    assert set(fm.keys()) == {"rouge", "truth_ratio", "privleak"}
    assert set(fd.keys()) == {"retain"}
    assert ftc["evaluation_mode"] == "unguided"


def test_filter_retain_guided_native_includes_privleak():
    m = {
        "probability": {},
        "extraction_strength": {},
        "truth_ratio": {},
        "privleak": {},
        "rouge": {},
    }
    data = {"retain": object()}
    tc = {"evaluation_mode": "unguided"}
    fm, fd, ftc = filter_metrics_and_data_for_pass("retain__guided_native", m, data, tc)
    assert set(fm.keys()) == {"probability", "extraction_strength", "truth_ratio", "privleak"}
    assert "privleak" in get_pass_spec("retain__guided_native").internal_metric_keys
    assert ftc["evaluation_mode"] == "guided_native"


def test_all_canonical_eight_pass_specs_exist():
    for pid in canonical_pass_ids_eight():
        get_pass_spec(pid)


def test_forget_guided_skew_mode():
    s = get_pass_spec("forget__guided_skew")
    assert s.evaluation_mode == "guided_skew"


def test_forget_guided_native_includes_golden_token_heatmap():
    s = get_pass_spec("forget__guided_native")
    assert "golden_token_prob_heatmap" in s.internal_metric_keys
    assert "trajectory_forget_golden_token_prob_heatmap" in s.display_names_emitted


def test_implemented_pass_specs_are_subset_of_canonical_twelve():
    """Skew variants extend the eight canonical legs; every spec must be discoverable."""
    twelve = set(canonical_pass_ids_twelve())
    for pid in list_all_pass_specs():
        assert pid in twelve
