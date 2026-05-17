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
