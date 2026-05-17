import os
from typing import Optional, Sequence

from rl_algo_impls.runner.evaluate import EvalArgs, evaluate_model
from rl_algo_impls.runner.running_utils import base_parser


def enjoy(argv: Optional[Sequence[str]] = None) -> None:
    parser = base_parser(multiple=True)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-episodes", type=int, default=3)
    parser.add_argument("--deterministic-eval", action="store_true", default=None)
    parser.add_argument("--stochastic-eval", action="store_true")
    parser.add_argument("--no-print-returns", action="store_true")
    parser.add_argument("--wandb-run-path", type=str, default=None)
    parser.add_argument("--video-path", type=str, default=None)
    parser.add_argument("--visualize-model-path", type=str, default=None)
    parser.add_argument("--thop", action="store_true")
    parser.add_argument("--tensorboard-folder", type=str, default=None)
    parsed = vars(parser.parse_args(argv))

    deterministic_eval = parsed.pop("deterministic_eval")
    if parsed.pop("stochastic_eval"):
        deterministic_eval = False
    parsed["deterministic_eval"] = deterministic_eval
    parsed["render"] = not parsed.pop("no_render")
    parsed["best"] = not parsed.pop("latest")

    for run_args in EvalArgs.expand_from_dict(parsed):
        evaluate_model(run_args, os.getcwd())


if __name__ == "__main__":
    enjoy()
