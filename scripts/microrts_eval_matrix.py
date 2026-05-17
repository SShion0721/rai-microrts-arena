import json
from dataclasses import asdict

from rl_algo_impls.microrts.eval_matrix import EvaluationSpec, build_eval_jobs


def main() -> None:
    spec = EvaluationSpec()
    print(
        json.dumps(
            {
                "spec": asdict(spec),
                "total_games": spec.total_games,
                "jobs": build_eval_jobs(spec),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
