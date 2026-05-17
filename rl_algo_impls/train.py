from typing import Optional, Sequence

from rl_algo_impls.runner.config import TrainArgs
from rl_algo_impls.runner.running_utils import base_parser
from rl_algo_impls.runner.train import train as run_train


def train(argv: Optional[Sequence[str]] = None) -> None:
    parser = base_parser(multiple=True)
    parser.add_argument("--wandb-project-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=[])
    parser.add_argument("--wandb-group", type=str, default=None)
    args_by_name = vars(parser.parse_args(argv))

    for args in TrainArgs.expand_from_dict(args_by_name):
        run_train(args)


if __name__ == "__main__":
    train()
