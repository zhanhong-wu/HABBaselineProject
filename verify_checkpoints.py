#!/usr/bin/env python3
"""Strictly verify legacy best-model weights and training checkpoints."""

import argparse
import gc
import json
import threading
from pathlib import Path

import torch
import torch.optim as optim

from hab_baseline.checkpoints import load_best_model, load_checkpoint
from hab_baseline.models import BASELINE_MODEL_NAMES, available_models, build_model, legacy_file_tag


def start_heartbeat(model_name):
    """Keep long remote verification sessions visibly active during large file reads."""
    stop = threading.Event()

    def report_progress():
        while not stop.wait(60):
            print(f"  [{model_name}] verification is still running...", flush=True)

    thread = threading.Thread(target=report_progress, daemon=True)
    thread.start()
    return stop, thread


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Directory containing legacy best_model_*.pth and checkpoint_*.pth files.")
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--mode", choices=("best", "checkpoint", "both"), default="both")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--in-channels", type=int, default=12)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--forward", action="store_true", help="Also run one CPU forward pass per model.")
    parser.add_argument("--report", type=Path, default=Path("checkpoint_verification.json"))
    return parser.parse_args()


def legacy_paths(root, model_name, patch_size):
    tag = legacy_file_tag(model_name)
    return {
        "best": root / f"best_model_patch{patch_size}_{model_name}_{tag}.pth",
        "checkpoint": root / f"checkpoint_patch{patch_size}_{model_name}_{tag}.pth",
    }


def main():
    args = parse_args()
    names = BASELINE_MODEL_NAMES if args.models == ["all"] else args.models
    unknown = set(names) - set(available_models())
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    report = {"root": str(args.root.resolve()), "torch_version": torch.__version__,
              "mode": args.mode, "models": {}}
    failed = False

    for name in names:
        print(f"\n[{name}] Building model and loading parameters...", flush=True)
        result = {"best": None, "checkpoint": None, "forward": None, "error": None}
        model = optimizer = output = None
        stop_heartbeat, heartbeat_thread = start_heartbeat(name)
        try:
            model = build_model(name, args.in_channels, args.num_classes)
            paths = legacy_paths(args.root, name, args.patch_size)
            if args.mode in ("best", "both"):
                result["best"] = load_best_model(model, paths["best"], map_location="cpu", strict=True)
                print(f"  BEST OK: {result['best']['parameters']} tensors", flush=True)
            if args.mode in ("checkpoint", "both"):
                optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
                result["checkpoint"] = load_checkpoint(model, optimizer, paths["checkpoint"],
                                                       map_location="cpu", strict=True)
                print(f"  CHECKPOINT OK: next_epoch={result['checkpoint']['next_epoch']}, "
                      f"optimizer_states={result['checkpoint']['optimizer_state_entries']}", flush=True)
            if result["best"] and result["checkpoint"]:
                assert result["best"]["parameters"] == result["checkpoint"]["parameters"]
            if args.forward:
                model.eval()
                with torch.no_grad():
                    output = model(torch.zeros(1, args.in_channels, args.patch_size, args.patch_size))
                result["forward"] = list(output.shape)
                print(f"  FORWARD OK: {result['forward']}", flush=True)
        except Exception as exc:
            failed = True
            result["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {result['error']}", flush=True)
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join()
        report["models"][name] = result
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        del model, optimizer, output
        gc.collect()

    report["status"] = "failed" if failed else "passed"
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nVerification {report['status'].upper()}: {args.report.resolve()}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
