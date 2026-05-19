from rl_algo_impls.wrappers.league_training_wrapper import (
    LeagueConfig,
    MatchupStats,
    PBTConfig,
)


def test_matchup_stats_score_rate():
    stats = MatchupStats()
    stats.update(1.0)
    stats.update(0.5)
    stats.update(0.0)

    assert stats.matches == 3
    assert stats.score_rate == 0.5


def test_league_config_accepts_pbt_dict():
    config = LeagueConfig(priority_mode="pfsp", pbt={"enabled": True})

    assert config.priority_mode == "pfsp"
    assert isinstance(config.pbt, PBTConfig)
    assert config.pbt.enabled


def test_league_config_tracks_population_roles():
    config = LeagueConfig(
        priority_mode="pfsp",
        role_by_policy_idx={0: "main_exploiter"},
        role_sampling_weights={"main_exploiter": 2.0, "league": 1.0},
        recent_result_window=3,
    )

    assert config.role_by_policy_idx[0] == "main_exploiter"
    assert config.role_sampling_weights["main_exploiter"] == 2.0
    assert config.recent_result_window == 3
