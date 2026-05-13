# Copyright (c) 2026 ax-remote-infer authors. Licensed under BSD-3-Clause.
"""
Remote NPU inference demo using RemoteAXExecutionProvider.

Usage:
    # auto-discover (assumes one device on the LAN)
    python remote_classification.py -m mobilenetv2.axmodel -i cat.jpg

    # manual
    python remote_classification.py -m mobilenetv2.axmodel -i cat.jpg --host 192.168.1.42

This is a near-copy of classification.py with two changes:
  - InferenceSession is constructed with the RemoteAXExecutionProvider
  - discover_devices() shows what's on the LAN
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import axengine as axe


def load_image(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    return arr[None, ...]  # NHWC, batch=1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", required=True)
    p.add_argument("-i", "--image", required=True)
    p.add_argument("--host", default=None,
                   help="device IP. If omitted, auto-discover via UDP broadcast.")
    p.add_argument("--port", type=int, default=18500)
    p.add_argument("--device-id", type=int, default=-1,
                   help="AXCL device id (only meaningful for AXCL hosts).")
    p.add_argument("--remote-provider", default=None,
                   help="Hint for the device daemon: AxEngineExecutionProvider "
                        "or AXCLRTExecutionProvider. Default: auto.")
    p.add_argument("--repeat", type=int, default=5)
    args = p.parse_args()

    host = args.host
    if host is None:
        print("[INFO] discovering devices on LAN (UDP broadcast, 2s)...")
        devs = axe.discover_devices(timeout=2.0)
        if not devs:
            print("[ERROR] no devices found; pass --host explicitly.", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] found {len(devs)} device(s):")
        for d in devs:
            print(f"  - {d.ip}:{d.tcp_port}  chip={d.chip_type} backend={d.backend} host={d.hostname}")
        host = devs[0].ip
        args.port = devs[0].tcp_port
        print(f"[INFO] picking {host}:{args.port}")

    provider_options = {"host": host, "port": args.port, "device_id": args.device_id}
    if args.remote_provider:
        provider_options["remote_provider"] = args.remote_provider

    with axe.InferenceSession(
        args.model,
        providers=[("RemoteAXExecutionProvider", provider_options)],
    ) as sess:
        meta_in = sess.get_inputs()[0]
        meta_out = sess.get_outputs()
        # MobileNetV2 takes NHWC uint8 typically; detect size from meta.
        size = int(meta_in.shape[1])
        x = load_image(args.image, size)
        if list(x.shape) != list(meta_in.shape):
            print(f"[WARN] reshaping input from {x.shape} to {meta_in.shape}")
            x = x.reshape(meta_in.shape).astype(meta_in.dtype, copy=False)
        if x.dtype != meta_in.dtype:
            x = x.astype(meta_in.dtype, copy=False)

        # Warmup
        sess.run(None, {meta_in.name: x})

        times = []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            outs = sess.run(None, {meta_in.name: x})
            times.append(time.perf_counter() - t0)

        scores = outs[0].reshape(-1).astype(np.float32)
        # top-5
        order = np.argsort(scores)[-5:][::-1]
        print("\n  Top 5 predictions:")
        for idx in order:
            print(f"    class={int(idx)}  score={float(scores[idx]):.4f}")
        print(f"\n  end-to-end (incl. transport): min/avg/max = "
              f"{min(times) * 1000:.2f} / {sum(times) / len(times) * 1000:.2f} / "
              f"{max(times) * 1000:.2f} ms ({len(times)} runs)")


if __name__ == "__main__":
    main()
