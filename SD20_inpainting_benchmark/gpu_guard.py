import argparse
import subprocess
import time


def gpu_state():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    rows = []
    for line in result.stdout.strip().splitlines():
        index, name, total, used, free, utilization = [part.strip() for part in line.split(",")]
        rows.append({"index": int(index), "name": name, "total_mib": int(total), "used_mib": int(used), "free_mib": int(free), "utilization_percent": int(utilization)})
    if len(rows) != 1:
        raise RuntimeError("This benchmark guard is configured for the current single-GPU host only")
    return rows[0]


def wait_for_memory(required_mib=8192, timeout=0, interval=30):
    start = time.monotonic()
    while True:
        state = gpu_state()
        if state["free_mib"] >= required_mib:
            print(f"GPU admission: {state}", flush=True)
            return state
        if timeout and time.monotonic() - start >= timeout:
            raise TimeoutError(f"GPU memory unavailable: need {required_mib} MiB, have {state['free_mib']} MiB")
        print(f"Waiting for GPU memory: need {required_mib} MiB, have {state['free_mib']} MiB; existing jobs are untouched", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--required-mib", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=0)
    args = parser.parse_args()
    if args.required_mib < 1 or args.timeout < 0:
        parser.error("Invalid admission threshold or timeout")
    wait_for_memory(args.required_mib, args.timeout)
