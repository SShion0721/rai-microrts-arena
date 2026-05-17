import os
from typing import Optional, Sequence

from rl_algo_impls.runner.running_utils import base_parser
from rl_algo_impls.runner.selfplay_evaluate import SelfplayEvalArgs, selfplay_evaluate


def selfplay_enjoy(argv: Optional[Sequence[str]] = None) -> None:
    parser = base_parser(multiple=True)
    parser.add_argument("--wandb-run-paths", type=str, nargs="*", default=[])
    parser.add_argument("--model-file-paths", type=str, nargs="*", default=[])
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--deterministic-eval", action="store_true", default=None)
    parser.add_argument("--stochastic-eval", action="store_true")
    parser.add_argument("--no-print-returns", action="store_true")
    parser.add_argument("--video-path", type=str, default=None)
    parsed = vars(parser.parse_args(argv))

    deterministic_eval = parsed.pop("deterministic_eval")
    if parsed.pop("stochastic_eval"):
        deterministic_eval = False
    parsed["deterministic_eval"] = deterministic_eval
    parsed["best"] = not parsed.pop("latest")

    for run_args in SelfplayEvalArgs.expand_from_dict(parsed):
        selfplay_evaluate(run_args, os.getcwd())


if __name__ == "__main__":
    selfplay_enjoy()
