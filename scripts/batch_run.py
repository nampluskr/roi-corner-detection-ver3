# scripts/batch_run.py: batch runner for train via subprocess

import argparse
import importlib.util
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config import get_exp_name, get_output_dir

DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "scripts", "batch_config.py")
RUN_MODES = ["train", "evaluate", "predict"]
MODES = RUN_MODES + ["all"]

PASS_KEYS = [
    "dataset", "csv_path", "network", "head", "device", "batch_size", "max_epochs",
    "num_workers", "train_size", "valid_size", "test_size", "checkpoint", "output_dir",
    "warmup_epochs",
]


def resolve_config_path(path=None):
    """Resolve a config file path from the working directory or project root."""
    if path is None:
        return DEFAULT_CONFIG_PATH
    if os.path.isabs(path):
        candidates = [path]
    else:
        candidates = [os.path.abspath(path), os.path.join(PROJECT_ROOT, path)]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError("batch config file not found: %s" % path)


def load_configs(path=None):
    """Load and return CONFIGS from a Python config file."""
    config_path = resolve_config_path(path)
    spec = importlib.util.spec_from_file_location("_batch_run_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create module spec for batch config: %s" % config_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise RuntimeError("failed to load batch config %s: %s" % (config_path, error)) from error
    if not hasattr(module, "CONFIGS"):
        raise ValueError("batch config must define CONFIGS: %s" % config_path)
    configs = module.CONFIGS
    if not isinstance(configs, (list, tuple)):
        raise TypeError("CONFIGS must be a list or tuple: %s" % config_path)
    return list(configs)


def get_cli_args(cfg, mode):
    """Return CLI args for one mode script built from one config dict."""
    args = ["--model", cfg["model"]]
    if mode == "train":
        args.append("--save")
    for key in PASS_KEYS:
        if key in cfg:
            value = cfg[key]
            if isinstance(value, (list, tuple)):
                args += ["--%s" % key] + [str(item) for item in value]
            else:
                args += ["--%s" % key, str(value)]
    if mode in ("evaluate", "predict") and not cfg.get("checkpoint"):
        output_dir = cfg.get("output_dir") or get_output_dir(cfg)
        args += ["--checkpoint", os.path.join(output_dir, "model.pth")]
    return args


def run(mode, configs):
    """Run one mode script for each config via subprocess and report a summary."""
    script = os.path.join("scripts", "%s.py" % mode)
    total = len(configs)
    results = []

    for i, cfg in enumerate(configs, 1):
        exp_name = get_exp_name(cfg)
        print("\n[%d/%d] %s | %s" % (i, total, mode, exp_name))
        cmd = [sys.executable, script] + get_cli_args(cfg, mode)
        try:
            subprocess.run(cmd, check=True)
            results.append({"exp_name": exp_name, "success": True, "error": None})
            print("[OK] %s" % exp_name)
        except subprocess.CalledProcessError as e:
            results.append({"exp_name": exp_name, "success": False, "error": str(e)})
            print("[FAIL] %s: %s" % (exp_name, e))

    success = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    print("\n[done] %s: %d/%d success, %d failed" % (mode, len(success), total, len(failed)))
    for r in failed:
        print("  [FAIL] %s: %s" % (r["exp_name"], r["error"]))
    return results


def main():
    """Parse runner args and execute configured experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, default="train")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Python file defining CONFIGS (default: scripts/batch_config.py)")
    args = parser.parse_args()

    try:
        configs = load_configs(args.config)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))

    modes = RUN_MODES if args.mode == "all" else [args.mode]
    results = []
    for mode in modes:
        results.extend(run(mode, configs))
    if any(not result["success"] for result in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
