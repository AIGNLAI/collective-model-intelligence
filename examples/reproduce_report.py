"""Re-evaluate pinned public predictions or refit the report's public baselines.

Run from a repository clone using ``python -m examples.reproduce_report``.
Replay includes Mio; public baseline refitting does not expose private Mio training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from benchmark.data import DEFAULT_REVISION, load_robust_route_bench, materialize_routing_split
from benchmark.data.huggingface import DATASET_ID
from evaluation.report_release import (
    METHOD_NAMES,
    SCENARIO_SEEDS,
    assert_report_matches,
    build_report_results,
)
from evaluation.report_reproduction import evaluate_saved_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("replay", "fit"), default="replay")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--scenarios", nargs="+", choices=tuple(SCENARIO_SEEDS), default=list(SCENARIO_SEEDS)
    )
    parser.add_argument(
        "--methods", nargs="+", choices=tuple(METHOD_NAMES), default=list(METHOD_NAMES)
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from huggingface_hub import HfApi, hf_hub_download

    # A pinned commit already identifies immutable files and needs no API lookup.
    revision = (
        args.revision
        if re.fullmatch(r"[0-9a-f]{40}", args.revision)
        else HfApi().dataset_info(DATASET_ID, revision=args.revision).sha
    )

    def download(relative):
        return Path(hf_hub_download(DATASET_ID, relative, repo_type="dataset", revision=revision))

    audit = json.loads(download("release-audit.json").read_text())
    if not audit.get("release_integrity_verified"):
        raise ValueError("Dataset release has not passed integrity validation")
    manifest = json.loads(download("report/reproduction-manifest.json").read_text())
    expected = json.loads(download("report/aggregate-results.json").read_text())
    args.output.mkdir(parents=True)
    results = {}
    for scenario in args.scenarios:
        data = {
            split: materialize_routing_split(
                load_robust_route_bench(scenario, split, revision=revision),
                scenario=scenario,
                split=split,
            )
            for split in ("test", "validation")
        }
        results[scenario] = {method: {} for method in args.methods}
        for seed in SCENARIO_SEEDS[scenario]:
            run_root = args.output / scenario / f"seed{seed}"
            if args.mode == "fit":
                methods = [method for method in args.methods if method != "mio"]
                if methods:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "examples.run_report_baselines",
                            "--scenario",
                            scenario,
                            "--seed",
                            str(seed),
                            "--device",
                            args.device,
                            "--revision",
                            revision,
                            "--output",
                            str(run_root),
                            "--methods",
                            *methods,
                        ],
                        check=True,
                    )
            run_root.mkdir(parents=True, exist_ok=True)
            for method in args.methods:
                # Mio is replay-only, explicitly even when refitting public baselines.
                if args.mode == "fit" and method != "mio":
                    path = run_root / f"{method}.npz"
                else:
                    artifact = manifest["scenarios"][scenario]["methods"][method][str(seed)]
                    path = download(artifact["path"])
                    if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
                        raise ValueError("Published prediction checksum mismatch")
                card = evaluate_saved_predictions(
                    path,
                    data["test"],
                    data["validation"],
                    cost_unit="CNY" if scenario == "chinese" else "USD",
                )
                card["prediction_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                card["data_revision"] = revision
                card["training_reproduced"] = args.mode == "fit" and method != "mio"
                if scenario == "multimodal":
                    card["feature_dimension"] = data["test"].embeddings.shape[1]
                    card["feature_provenance"] = json.loads(
                        str(data["test"].metadata[0]["provenance_json"])
                    )["embedding"]
                results[scenario][method][seed] = card
                (run_root / f"{method}-replay.json").write_text(json.dumps(card, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "scenario": scenario,
                            "method": method,
                            "seed": seed,
                            "training_reproduced": card["training_reproduced"],
                        }
                    ),
                    flush=True,
                )
    if set(args.scenarios) == set(SCENARIO_SEEDS) and set(args.methods) == set(METHOD_NAMES):
        aggregate = build_report_results(results, data_revision=revision)
        if args.mode == "replay":
            assert_report_matches(aggregate, expected)
        (args.output / "aggregate-results.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    (args.output / "run.json").write_text(
        json.dumps(
            {
                "data_revision": revision,
                "mode": args.mode,
                "mio_training_reproduced": False,
                "scenarios": args.scenarios,
                "methods": args.methods,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
