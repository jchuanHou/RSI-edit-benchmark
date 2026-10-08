import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from common import DEFAULT_DATA, ROOT, atomic_json, exclusive_lock, now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "sd20_inpainting_unipc_s42")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--limit", type=int, default=0, help="Nonzero: smoke run only, reported as partial")
    parser.add_argument("--skip-fwd", action="store_true", help="For lightweight smoke testing only")
    args = parser.parse_args()
    if args.limit < 0 or (args.skip_fwd and not args.limit):
        parser.error("Negative limit or skipping FWD on a full benchmark is not allowed")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    logs = args.run_dir / "logs"
    logs.mkdir(exist_ok=True)
    state_path = args.run_dir / "job_status.json"
    state = {"pid": os.getpid(), "started_at": now(), "status": "running", "limit": args.limit}
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONNOUSERSITE="1", TORCH_HOME=str(ROOT / "cache" / "torch"))
    child = None

    def stop(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with exclusive_lock(args.run_dir / ".job.lock"):
        try:
            for phase, script in (("inference", "infer.py"), ("evaluation", "evaluate.py")):
                state.update(phase=phase, updated_at=now())
                atomic_json(state_path, state)
                command = [sys.executable, "-u", str(ROOT / script), "--run-dir", str(args.run_dir.resolve()), "--dataset", str(args.dataset.resolve())]
                if args.limit:
                    command += ["--limit", str(args.limit)] if phase == "inference" else ["--allow-partial"]
                if args.skip_fwd and phase == "evaluation":
                    command.append("--skip-fwd")
                print(f"{now()} Starting {phase}", flush=True)
                with (logs / f"{phase}.log").open("a") as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    state.update(child_pid=child.pid)
                    atomic_json(state_path, state)
                    returncode = child.wait()
                    child = None
                if returncode:
                    state.update(status="failed", exit_code=returncode, finished_at=now())
                    atomic_json(state_path, state)
                    return returncode
            report = json.loads((args.run_dir / "eval_results.json").read_text())
            expected_status = "partial" if args.limit and report["evaluated"] < report["expected"] else "complete"
            if report["status"] != expected_status:
                raise RuntimeError(f"Unexpected evaluation status: {report['status']}")
            state.update(status="smoke_complete" if args.limit else "complete", finished_at=now(), evaluated=report["evaluated"], exit_code=0)
            atomic_json(state_path, state)
            print(f"{now()} {state['status']}: {report['evaluated']}/{report['expected']} samples", flush=True)
            return 0
        except BaseException as exc:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", finished_at=now(), error=f"{type(exc).__name__}: {exc}", exit_code=1)
            atomic_json(state_path, state)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
