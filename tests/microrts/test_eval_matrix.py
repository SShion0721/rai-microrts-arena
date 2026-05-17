from rl_algo_impls.microrts.eval_matrix import EvaluationSpec, WinRateEstimate, build_eval_jobs


def test_eval_matrix_counts_both_player_sides():
    spec = EvaluationSpec(maps=("m1", "m2"), opponents=("bot",), games_per_side=3)

    assert spec.total_games == 12
    assert len(build_eval_jobs(spec)) == 4


def test_wilson_interval_is_bounded():
    low, high = WinRateEstimate(wins=8, draws=1, losses=1).wilson_interval()

    assert 0 <= low <= high <= 1
