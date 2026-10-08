import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from common import DEFAULT_DATA, ROOT, atomic_json, exclusive_lock, now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "brushnet_sd15_random_s42")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    logs = args.run_dir / "logs"
    logs.mkdir(exist_ok=True)
    state_path = args.run_dir / "job_status.json"
    state = {"pid": os.getpid(), "started_at": now(), "status": "running"}
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
    with exclusive_lock(args.run_dir / ".job.lock"):
        try:
            for phase, script in (("inference", "infer.py"), ("evaluation", "evaluate.py")):
                state.update(phase=phase, updated_at=now())
                atomic_json(state_path, state)
                command = [sys.executable, "-u", str(ROOT / script), "--run-dir", str(args.run_dir.resolve()), "--dataset", str(args.dataset.resolve())]
                print(f"{now()} Starting {phase}", flush=True)
                with (logs / f"{phase}.log").open("a") as log:
                    result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    state.update(status="failed", exit_code=result.returncode, finished_at=now())
                    atomic_json(state_path, state)
                    return result.returncode
            report = json.loads((args.run_dir / "eval_results.json").read_text())
            if report["status"] != "complete":
                raise RuntimeError("Evaluation did not finish completely")
            state.update(status="complete", finished_at=now(), evaluated=report["evaluated"], exit_code=0)
            atomic_json(state_path, state)
            print(f"{now()} COMPLETE: {report['evaluated']} samples", flush=True)
            return 0
        except Exception as exc:
            state.update(status="failed", finished_at=now(), error=f"{type(exc).__name__}: {exc}", exit_code=1)
            atomic_json(state_path, state)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
