# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prepare data and the gRNAde environment.

python prepare.py rnainvbench_eterna100_v1 --output data/eterna100_v1.jsonl --limit 20
python prepare.py --setup ~/rna_agent_env      # gRNAde checkout, model weights, and a Python env for it
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


SUITES_DIR = Path(__file__).resolve().parent / "suites"


def task_description(target: dict) -> str:
    """Placeholder input. The campaign builds its own prompt, so the model does not read this."""
    lines = [f"Design an RNA sequence of length {target['length']} for this target structure:"]
    lines.append(target["target_structure"])
    if target.get("fixed_prefix"):
        lines.append(f"Fixed prefix: {target['fixed_prefix']}")
    return "\n".join(lines)


def build_rows(suite_name: str, limit: int = 0, seed: int = 0) -> list[dict]:
    path = SUITES_DIR / f"{suite_name}.yaml"
    if not path.is_file():
        known = sorted(p.stem for p in SUITES_DIR.glob("*.yaml"))
        raise SystemExit(f"unknown suite {suite_name!r}; choose from {known}")
    suite = yaml.safe_load(path.read_text())
    targets = list(suite["targets"])
    if limit:
        random.Random(seed).shuffle(targets)
        targets = targets[:limit]
    return [
        {
            "responses_create_params": {"input": [{"role": "user", "content": task_description(t)}]},
            "suite": suite["name"],
            "target": t,
        }
        for t in targets
    ]


# What gRNAde imports. torch-scatter and torch-cluster are left out: the agent ships pure PyTorch stand-ins.
GRNADE_PACKAGES = (
    "numpy scipy pandas pyyaml tqdm einops ml_collections python-dotenv torchmetrics biopython biotite mdanalysis "
    "torch-geometric arnie wandb gdown draw_rna cpdb-protein huggingface_hub"
).split()


def setup(home: Path) -> None:
    """Check out gRNAde, download the checkpoints it loads (public, no login) and build a Python env for it.

    Afterwards run the agent with `GRNADE_REPO=<home>/geometric-rna-design` and `<home>/venv/bin` first on PATH.
    """
    from huggingface_hub import snapshot_download

    home.mkdir(parents=True, exist_ok=True)
    repo = home / "geometric-rna-design"
    if not repo.exists():
        url = "https://github.com/chaitjo/geometric-rna-design.git"
        subprocess.run(["git", "clone", "--depth", "1", url, str(repo)], check=True)
    # design.py loads all three models from checkpoints/ inside the checkout.
    patterns = ["gRNAde_drop3d@0.75_maxlen@500.h5", "ribonanzanet/*", "ribonanzanet_sec_struct/*", "config.yaml"]
    snapshot_download("chaitjo/gRNAde", local_dir=repo / "checkpoints", allow_patterns=patterns)
    shutil.rmtree(repo / "checkpoints" / ".cache", ignore_errors=True)

    venv = home / "venv"
    if not venv.exists():
        # The venv is built from the interpreter running this script.
        if sys.version_info < (3, 10):
            raise SystemExit(f"run prepare.py with Python 3.10 or newer, not {sys.version.split()[0]}")
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    pip = [str(venv / "bin" / "python"), "-m", "pip", "install", "--quiet"]
    # The CUDA 12.8 wheel on Linux, the default wheel elsewhere.
    index = ["--index-url", "https://download.pytorch.org/whl/cu128"] if sys.platform == "linux" else []
    subprocess.run([*pip, "torch", *index], check=True)
    subprocess.run([*pip, *GRNADE_PACKAGES], check=True)
    print(f"ready: GRNADE_REPO={repo}  python={venv / 'bin' / 'python'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suite", nargs="?", help="a file stem under suites/")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--setup", type=Path, metavar="DIR", help="set up gRNAde and its weights in DIR")
    parser.add_argument("--limit", type=int, default=0, help="a seeded random subset of this many targets")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.setup:
        return setup(args.setup)
    if not (args.suite and args.output):
        parser.error("give a suite and --output, or --setup DIR")

    rows = build_rows(args.suite, args.limit, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
