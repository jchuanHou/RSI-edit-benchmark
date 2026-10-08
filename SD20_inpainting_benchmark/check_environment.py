import importlib
import importlib.metadata
import sys
from pathlib import Path

from common import ROOT, atomic_json, now, runtime_versions


def main():
    import torch
    import diffusers
    from diffusers import StableDiffusionInpaintPipeline

    expected = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if "==" in line and not line.startswith("-"):
            name, version = line.split("==", 1)
            expected[name] = version
    for name, version in expected.items():
        actual = importlib.metadata.version(name)
        if actual != version:
            raise RuntimeError(f"{name}: expected {version}, found {actual}")
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise RuntimeError("Use this project's .venv/bin/python")
    package = Path(diffusers.__file__).resolve()
    if (ROOT / ".venv").resolve() not in package.parents:
        raise RuntimeError(f"diffusers must be installed in this project's environment: {package}")
    if hasattr(diffusers, "BrushNetModel") or hasattr(diffusers, "StableDiffusionBrushNetPipeline"):
        raise RuntimeError("Custom BrushNet diffusers is not allowed")
    pipeline_module = importlib.import_module(StableDiffusionInpaintPipeline.__module__)
    if Path(pipeline_module.__file__).resolve().is_relative_to(package.parent) is False:
        raise RuntimeError("Unexpected pipeline module")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    report = {
        "checked_at": now(), "python": sys.version, "executable": sys.executable,
        "versions": runtime_versions(), "diffusers_path": str(package),
        "pipeline": StableDiffusionInpaintPipeline.__module__ + "." + StableDiffusionInpaintPipeline.__name__,
        "torch_path": torch.__file__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "user_site_disabled": bool(sys.flags.no_user_site),
        "environment_policy": "Independent project venv; system-site-packages disabled; no sibling project packages",
    }
    atomic_json(ROOT / "environment.json", report)
    print(f"Environment OK: official diffusers {diffusers.__version__}; {report['gpu']}")


if __name__ == "__main__":
    main()
