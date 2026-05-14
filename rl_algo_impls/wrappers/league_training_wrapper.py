import logging
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Generic, List, Optional, Tuple

import numpy as np

from rl_algo_impls.wrappers.self_play_wrapper import SelfPlayWrapper, ObsType


@dataclass
class LeaguePolicyStats:
    """ELO-style rating tracking for a policy in the league pool."""

    rating: float = 1500.0
    matches: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def win_rate(self) -> float:
        if self.matches == 0:
            return 0.5
        return self.wins / self.matches

    def update_elo(
        self, opponent_rating: float, result: float, k: float = 32.0
    ) -> None:
        """Update ELO rating.

        Args:
            opponent_rating: Opponent's current rating
            result: 1.0 (win), 0.5 (draw), 0.0 (loss)
            k: K-factor for rating sensitivity
        """
        expected = 1.0 / (1.0 + 10.0 ** ((opponent_rating - self.rating) / 400.0))
        self.rating += k * (result - expected)
        self.matches += 1
        if result > 0.75:
            self.wins += 1
        elif result > 0.25:
            self.draws += 1
        else:
            self.losses += 1


@dataclass
class LeagueConfig:
    """Configuration for league training opponent selection."""

    # Prioritization mode: "hardest" | "closest" | "random"
    priority_mode: str = "closest"

    # Target win rate range for "closest" mode (policies with win rate in this range are preferred)
    target_win_rate_min: float = 0.3
    target_win_rate_max: float = 0.7

    # ELO K-factor
    elo_k: float = 32.0

    # Minimum matches before a policy's ELO is considered reliable
    min_matches_for_elo: int = 5

    # Exploration rate: probability of random opponent selection (epsilon-greedy)
    exploration_rate: float = 0.1

    # Whether to track results automatically via env info
    auto_track_results: bool = True


class LeagueTrainingWrapper(SelfPlayWrapper, Generic[ObsType]):
    """Extends SelfPlayWrapper with ELO-based prioritized opponent selection.

    Instead of randomly swapping in old policies, this selects opponents that:
    - Are closest to the learner's current skill level (closest mode)
    - Present the hardest challenge (hardest mode)
    - Are randomly selected with epsilon-greedy exploration

    This is a simplified version of AlphaStar's league training / PFSP
    (Prioritized Fictitious Self-Play).
    """

    def __init__(
        self,
        env,
        config,
        num_old_policies: int = 0,
        save_steps: int = 20_000,
        swap_steps: int = 10_000,
        window: int = 10,
        swap_window_size: int = 2,
        selfplay_bots: Optional[Dict[str, Any]] = None,
        bot_always_player_2: bool = False,
        first_window_orig_policy: bool = False,
        league_config: Optional[LeagueConfig] = None,
    ) -> None:
        super().__init__(
            env,
            config,
            num_old_policies=num_old_policies,
            save_steps=save_steps,
            swap_steps=swap_steps,
            window=window,
            swap_window_size=swap_window_size,
            selfplay_bots=selfplay_bots,
            bot_always_player_2=bot_always_player_2,
            first_window_orig_policy=first_window_orig_policy,
        )
        self.league_config = league_config or LeagueConfig()
        self._league_stats: Dict[int, LeaguePolicyStats] = {}
        self._recent_results: Deque[Tuple[int, int, float]] = deque(maxlen=1000)

        logging.info(
            f"LeagueTrainingWrapper: priority_mode={self.league_config.priority_mode}, "
            f"exploration_rate={self.league_config.exploration_rate}"
        )

    def swap_policy(self, idx: int, swap_window_size: int = 1) -> None:
        """Override to use prioritized opponent selection instead of random."""
        if len(self.policies) <= 1 or random.random() < self.league_config.exploration_rate:
            # Fall back to random selection for exploration
            return super().swap_policy(idx, swap_window_size)

        selected_policy = self._prioritized_select_opponent()
        if selected_policy is None:
            return super().swap_policy(idx, swap_window_size)

        idx = idx // 2 * 2
        for j in range(swap_window_size * 2):
            if self.policy_assignments[idx + j]:
                self.policy_assignments[idx + j] = selected_policy
        self.steps_since_swap[idx : idx + swap_window_size * 2] = np.zeros(
            swap_window_size * 2
        )

    def _prioritized_select_opponent(self):
        """Select opponent from the policy pool based on priority mode.

        Returns:
            Selected Policy or None (delegate to random in caller)
        """
        from rl_algo_impls.shared.policy.policy import Policy

        # Get all policies in the pool with their indices
        policies_with_idx = list(enumerate(self.policies))
        if not policies_with_idx:
            return None

        mode = self.league_config.priority_mode
        if mode == "hardest":
            # Pick the policy with the highest ELO rating
            best_idx = max(
                policies_with_idx,
                key=lambda x: self._get_elo(x[0]),
            )[0]
            return self.policies[best_idx]
        elif mode == "closest":
            # Pick a policy with win rate closest to 50% against learner
            # (within target range)
            target_candidates = [
                (i, stats)
                for i, stats in self._league_stats.items()
                if stats.matches >= self.league_config.min_matches_for_elo
                and self.league_config.target_win_rate_min
                <= stats.win_rate
                <= self.league_config.target_win_rate_max
            ]
            if target_candidates:
                best_idx = min(
                    target_candidates,
                    key=lambda x: abs(x[1].win_rate - 0.5),
                )[0]
                return self.policies[best_idx]
        # Default: random
        return None

    def _get_elo(self, policy_idx: int) -> float:
        """Get ELO rating for a policy, returning default if not tracked."""
        if policy_idx in self._league_stats:
            return self._league_stats[policy_idx].rating
        return 1500.0

    def _track_results_from_info(self, info: list) -> None:
        """Extract win/loss results from env info dicts and update ELO.

        Attempts to read win-loss data from each env's info dict (e.g.,
        microRTS info["results"]["WinLoss"] or info["microrts_results"]).
        Only tracks results for envs where the opponent is an old policy
        from the pool (not a scripted bot).
        """
        if not self.league_config.auto_track_results:
            return

        for env_idx, env_info in enumerate(info):
            if not isinstance(env_info, dict):
                continue

            # Check what policy is assigned to this env pair
            pair_idx = env_idx // 2 * 2
            opponent_idx = pair_idx + (1 if env_idx % 2 == 0 else 0)
            opponent_policy = (
                self.policy_assignments[opponent_idx]
                if opponent_idx < len(self.policy_assignments)
                else None
            )
            if opponent_policy is None:
                continue

            # Find opponent's index in the deque
            try:
                opp_deque_idx = next(
                    i for i, p in enumerate(self.policies) if p is opponent_policy
                )
            except StopIteration:
                continue

            # Try to extract result from info
            result = None
            if "results" in env_info and isinstance(env_info["results"], dict):
                wl = env_info["results"].get("WinLoss")
                if wl is not None:
                    if wl > 0:
                        result = 0.0  # learner lost (opponent won)
                    elif wl < 0:
                        result = 1.0  # learner won
                    else:
                        result = 0.5  # draw

            if result is not None:
                # Learner is at a virtual index for ELO tracking
                learner_idx = -1
                self.track_result(learner_idx, opp_deque_idx, result)

    def track_result(
        self, policy_idx: int, opponent_idx: int, result: float
    ) -> None:
        """Track a match result between two policies.

        Args:
            policy_idx: Index of learner/active policy in pool (-1 for current learner)
            opponent_idx: Index of opponent policy in pool
            result: 1.0 (learner win), 0.5 (draw), 0.0 (learner loss)
        """
        # Only track the opponent's stats for ELO purposes
        # The learner gets tracked indirectly through opponent's view
        if opponent_idx not in self._league_stats:
            self._league_stats[opponent_idx] = LeaguePolicyStats()

        opponent = self._league_stats[opponent_idx]
        # Simple update: opponent's ELO moves based on result against fixed
        # learner rating of 1500
        opponent.update_elo(
            1500.0, result, k=self.league_config.elo_k
        )
        self._recent_results.append((policy_idx, opponent_idx, result))

    def step(self, actions: np.ndarray):
        """Override step to track match results for ELO updates."""
        obs, reward, terminations, truncations, info = super().step(actions)

        # Track results when episodes end
        if self.league_config.auto_track_results:
            env_info = (
                list(info) if isinstance(info, (tuple, list)) else info
            )
            if terminations.any() or truncations.any():
                self._track_results_from_info(env_info)

        return obs, reward, terminations, truncations, info

    @property
    def league_standings(self) -> List[Tuple[int, LeaguePolicyStats]]:
        """Return policy standings sorted by ELO rating (descending)."""
        return sorted(
            self._league_stats.items(), key=lambda x: x[1].rating, reverse=True
        )
