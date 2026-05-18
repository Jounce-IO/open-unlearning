"""Binding table and pass specs for multi-pass trajectory eval."""

from evals.metrics.trajectory_pass_binding import (
    DISPLAY_METRIC_BINDING,
    HM_AGGREGATE_SUBMETRIC_BINDING,
    canonical_pass_ids_eight,
    canonical_pass_ids_fourteen_mu,
    canonical_pass_ids_legacy_mu_norm,
    canonical_pass_ids_sixteen_mu,
    canonical_pass_ids_twelve,
    filter_metrics_and_data_for_pass,
    get_pass_spec,
    list_all_pass_specs,
    trajectory_pass_ids_extended,
)

_LEGACY_PASS_IDS = frozenset(
    {
        "forget__guided_native",
        "forget__guided_skew",
        "retain__guided_native",
        "retain__guided_skew",
        "ra__guided_native",
        "ra__guided_skew",
        "wf__guided_native",
        "wf__guided_skew",
    }
)


def test_display_metric_binding_count():
    assert len(DISPLAY_METRIC_BINDING) == 9


def test_hm_aggregate_submetric_binding_count():
    assert len(HM_AGGREGATE_SUBMETRIC_BINDING) == 9


def test_canonical_pass_ids_fourteen_mu():
    ids = canonical_pass_ids_fourteen_mu()
    assert len(ids) == 14
    assert ids == canonical_pass_ids_sixteen_mu()
    assert ids == canonical_pass_ids_eight() == canonical_pass_ids_twelve()
    assert "forget__unguided" in ids
    assert "retain__guided_tr_para" in ids
    assert "ra__guided_tr_correct" in ids
    assert "ra__guided_prob" not in ids
    assert "wf__guided_prob" not in ids
    assert "forget__guided_native" not in ids


def test_canonical_pass_ids_legacy_mu_norm():
    legacy = canonical_pass_ids_legacy_mu_norm()
    assert legacy == ("ra__guided_prob", "wf__guided_prob")
    for pid in legacy:
        get_pass_spec(pid)


def test_canonical_pass_ids_eight_is_fourteen_mu_alias():
    assert len(canonical_pass_ids_eight()) == 14


def test_canonical_pass_ids_twelve_is_fourteen_mu_alias():
    assert len(canonical_pass_ids_twelve()) == 14


def test_get_pass_spec_forget_unguided():
    s = get_pass_spec("forget__unguided")
    assert s.evaluation_mode == "unguided"
    assert "rouge" in s.internal_metric_keys
    assert "forget" in s.dataset_access_keys


def test_get_pass_spec_retain_guided_prob():
    s = get_pass_spec("retain__guided_prob")
    assert s.evaluation_mode == "guided_native"
    assert s.internal_metric_keys == frozenset({"probability"})
    assert "trajectory_retain_Q_A_Prob" in s.display_names_emitted


def test_get_pass_spec_ra_guided_tr_legs():
    assert "trajectory_ra_Q_A_Prob" in get_pass_spec("ra__guided_tr_correct").display_names_emitted
    assert "trajectory_ra_Q_A_PERT_Prob" in get_pass_spec("ra__guided_tr_pert").display_names_emitted


def test_filter_forget_unguided():
    m = {"rouge": {}, "probability": {}}
    data = {"forget": object(), "retain": object()}
    tc = {"evaluation_mode": "guided_native"}
    fm, fd, ftc = filter_metrics_and_data_for_pass("forget__unguided", m, data, tc)
    assert set(fm.keys()) == {"rouge"}
    assert set(fd.keys()) == {"forget"}
    assert ftc["evaluation_mode"] == "unguided"


def test_filter_retain_unguided():
    m = {"rouge": {}, "probability": {}}
    data = {"retain": object(), "forget": object()}
    tc = {"evaluation_mode": "guided_native"}
    fm, fd, ftc = filter_metrics_and_data_for_pass("retain__unguided", m, data, tc)
    assert set(fm.keys()) == {"rouge"}
    assert set(fd.keys()) == {"retain"}
    assert ftc["evaluation_mode"] == "unguided"


def test_filter_retain_sft_unguided():
    m = {"rouge": {}, "probability": {}}
    data = {"retain": object(), "forget": object()}
    tc = {"evaluation_mode": "guided_native"}
    fm, fd, ftc = filter_metrics_and_data_for_pass("retain_sft__unguided", m, data, tc)
    assert set(fm.keys()) == {"rouge"}
    assert set(fd.keys()) == {"retain"}
    assert ftc["evaluation_mode"] == "unguided"


def test_all_canonical_fourteen_pass_specs_exist():
    for pid in canonical_pass_ids_fourteen_mu():
        get_pass_spec(pid)


def test_forget_guided_skew_mode():
    s = get_pass_spec("forget__guided_skew")
    assert s.evaluation_mode == "guided_skew"


def test_forget_guided_native_includes_golden_token_heatmap():
    s = get_pass_spec("forget__guided_native")
    assert "golden_token_prob_heatmap" in s.internal_metric_keys
    assert "trajectory_forget_golden_token_prob_heatmap" in s.display_names_emitted


def test_implemented_pass_specs_are_discoverable():
    allowed = (
        set(canonical_pass_ids_fourteen_mu())
        | set(canonical_pass_ids_legacy_mu_norm())
        | set(trajectory_pass_ids_extended())
        | _LEGACY_PASS_IDS
    )
    for pid in list_all_pass_specs():
        assert pid in allowed
