from rl_algo_impls.microrts.eval_matrix import (
    EvaluationMetrics,
    EvaluationSpec,
    WinRateEstimate,
    build_eval_jobs,
)


def test_eval_matrix_counts_both_player_sides():
    spec = EvaluationSpec(maps=("m1", "m2"), opponents=("bot",), games_per_side=3)

    assert spec.total_games == 12
    assert len(build_eval_jobs(spec)) == 4


def test_wilson_interval_is_bounded():
    low, high = WinRateEstimate(wins=8, draws=1, losses=1).wilson_interval()

    assert 0 <= low <= high <= 1


def test_evaluation_metrics_dashboard_row_includes_system_metrics():
    metrics = EvaluationMetrics(
        win_rate=WinRateEstimate(wins=8, draws=1, losses=1),
        inference_ms_per_step=12.5,
        env_steps_per_second=256.0,
        policy_lag=2.0,
        invalid_action_rate=0.01,
    )

    row = metrics.dashboard_row()

    assert row["games"] == 10
    assert row["inference_ms_per_step"] == 12.5
    assert row["invalid_action_rate"] == 0.01
