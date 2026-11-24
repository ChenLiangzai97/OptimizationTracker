"""Synthetic one-device benchmark, one fresh process per variant."""

import argparse
import json
import resource
import statistics
import tempfile
import time

import torch

from .optimizer_tracker import OptimizationTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "full"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=220)
    args = parser.parse_args()
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
    device = torch.device(args.device)
    backend = getattr(torch, device.type) if device.type != "cpu" else None
    torch.manual_seed(7)
    model = torch.nn.ModuleList(
        [torch.nn.Linear(args.width, args.width) for _ in range(args.layers)]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0)
    x = torch.ones(4, args.width, device=device)
    samples = {"ordinary": [], "light": [], "heavy": []}
    peaks = {key: 0 for key in samples}
    with tempfile.TemporaryDirectory() as folder:
        tracker = OptimizationTracker(dict(model.named_parameters()), folder, device)
        for step in range(-4, args.steps + 1):
            if backend:
                backend.synchronize()
                backend.reset_peak_memory_stats(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            out = x
            for layer in model:
                out = layer(out)
            out.square().mean().backward()
            if args.mode == "full":
                tracker.before_update(step, optimizer)
            optimizer.step()
            if args.mode == "full":
                tracker.after_update(step, optimizer)
            if backend:
                backend.synchronize()
            if step > 0:
                kind = (
                    "heavy"
                    if step % 200 == 0
                    else "light"
                    if step % 20 == 0
                    else "ordinary"
                )
                samples[kind].append(time.perf_counter() - start)
                if backend:
                    peaks[kind] = max(peaks[kind], backend.max_memory_allocated(device))
        tracker.close()
    print(
        json.dumps(
            {
                "mode": args.mode,
                "device": str(device),
                "mean_ms": {
                    k: 1000 * statistics.mean(v) for k, v in samples.items() if v
                },
                "device_peak_bytes": peaks if backend else None,
                "process_peak_rss_kib_linux": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
            }
        )
    )


if __name__ == "__main__":
    main()
