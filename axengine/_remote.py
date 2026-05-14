# Copyright (c) 2026 ax-remote-infer authors. Licensed under BSD-3-Clause.
"""RemoteAXSession — pyaxengine InferenceSession backed by a TCP-attached
ax_remote_infer device daemon on the LAN.

Wire protocol mirror: see `_remote_wire`. Daemon source of truth:
ax-remote-infer/device_daemon/include/wire/wire.hpp.
"""

from __future__ import annotations

import os
import socket
import time
from typing import Any, Optional, Sequence

import numpy as np

from . import _remote_wire as wire
from ._base_session import Session, SessionOptions
from ._node import NodeArg


def _resolve_endpoint(provider_options: Optional[dict]) -> tuple[str, int]:
    if not provider_options:
        raise ValueError(
            "RemoteAXExecutionProvider requires provider_options with at least "
            "'host' set, e.g. providers=[('RemoteAXExecutionProvider', "
            "{'host':'192.168.1.42','port':18500})]")
    host = provider_options.get("host") or provider_options.get("ip")
    if not host:
        raise ValueError("provider_options must include 'host' (device IP)")
    port = int(provider_options.get("port", 18500))
    return str(host), port


def _provider_code(name: Optional[str]) -> int:
    if not name:
        return wire.PROVIDER_AUTO
    if name == "AxEngineExecutionProvider":
        return wire.PROVIDER_AXENGINE
    if name == "AXCLRTExecutionProvider":
        return wire.PROVIDER_AXCLRT
    return wire.PROVIDER_AUTO


def _load_bytes(path_or_bytes) -> bytes:
    if isinstance(path_or_bytes, bytes):
        return path_or_bytes
    if isinstance(path_or_bytes, (str, os.PathLike)):
        with open(path_or_bytes, "rb") as f:
            return f.read()
    raise TypeError(f"unsupported model source type: {type(path_or_bytes)}")


class RemoteAXSession(Session):
    """Connects to a remote ax_remote_infer daemon, ships the model bytes
    once, then forwards run() invocations over TCP.

    Per `run()` it prints four timings:
        in:     input transfer (client -> device)
        device: NPU inference (as measured on the device)
        out:    output transfer (device -> client)
        total:  end-to-end including framing

    Disable that line by setting AXINFER_QUIET=1.
    """

    def __init__(
        self,
        path_or_bytes,
        sess_options: Optional[SessionOptions] = None,
        provider_options: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self._endpoint = _resolve_endpoint(provider_options)
        self._device_id = int((provider_options or {}).get("device_id", -1))
        self._remote_provider = (provider_options or {}).get("remote_provider")
        self._timeout = float((provider_options or {}).get("timeout", 60.0))
        self._verbose = os.environ.get("AXINFER_QUIET", "0") != "1"

        self._sock: Optional[socket.socket] = None
        self._chip_type = ""
        self._engine_version = ""

        # last_timing is filled in by every run(). Fields:
        #   input_ms      : wall-clock to send the input tensors (client -> device)
        #   device_ms     : NPU inference time as reported by the device SDK
        #   output_ms     : derived: (recv_done - send_done) - device_ms; covers
        #                   server-side post-NPU bookkeeping + dev->client transfer
        #                   + client-side recv into Python
        #   total_ms      : send_start -> recv_done (does NOT include the
        #                   numpy.reshape/copy on the client; that's outside the
        #                   wire)
        #   input_bytes   : sum of input tensor sizes
        #   output_bytes  : sum of output tensor sizes (raw, on the wire)
        # See the README "timings" section for what each number includes.
        self.last_timing: dict = {}

        self._connect()
        self._hello()
        model_bytes = _load_bytes(path_or_bytes)
        self._load_model(model_bytes)

    # ---- connection / handshake ----

    def _connect(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self._timeout)
        s.connect(self._endpoint)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = s
        if self._verbose:
            print(f"[REMOTE] connected to {self._endpoint[0]}:{self._endpoint[1]}")

    def _hello(self) -> None:
        wire.send_frame(
            self._sock, wire.HELLO,
            wire.encode_hello(_provider_code(self._remote_provider), self._device_id))
        mtype, payload = wire.recv_frame(self._sock)
        if mtype == wire.ERROR:
            code, msg = wire.parse_error(payload)
            raise RuntimeError(f"remote refused HELLO: code={code} msg={msg}")
        if mtype != wire.HELLO_ACK:
            raise RuntimeError(f"unexpected reply to HELLO: type={mtype}")
        ack = wire.decode_hello_ack(payload)
        self._chip_type = ack["chip_type"]
        self._engine_version = ack["engine_version"]
        if self._verbose:
            print(f"[REMOTE] chip={ack['chip_type'] or '?'}  "
                  f"engine={ack['engine_version'] or '?'}  "
                  f"vnpu={ack['vnpu_type']}")

    def _load_model(self, model_bytes: bytes) -> None:
        t0 = time.perf_counter()
        wire.send_frame(self._sock, wire.LOAD_MODEL, wire.encode_load_model(model_bytes))
        mtype, payload = wire.recv_frame(self._sock)
        t1 = time.perf_counter()
        if mtype == wire.ERROR:
            code, msg = wire.parse_error(payload)
            raise RuntimeError(f"remote LOAD_MODEL failed: code={code} msg={msg}")
        if mtype != wire.MODEL_READY:
            raise RuntimeError(f"unexpected reply to LOAD_MODEL: type={mtype}")

        groups = wire.parse_model_ready(payload)
        self._shape_count = len(groups)
        self._inputs = [
            [NodeArg(d["name"], d["dtype"], d["shape"]) for d in g["inputs"]]
            for g in groups
        ]
        self._outputs = [
            [NodeArg(d["name"], d["dtype"], d["shape"]) for d in g["outputs"]]
            for g in groups
        ]
        # Remember per-group input metadata (with size_bytes) for fast RUN encoding.
        self._group_inputs_meta = [g["inputs"] for g in groups]
        self._group_outputs_meta = [g["outputs"] for g in groups]

        if self._verbose:
            n_in = sum(len(g["inputs"]) for g in groups)
            n_out = sum(len(g["outputs"]) for g in groups)
            print(f"[REMOTE] model loaded: {len(model_bytes)/1e6:.2f} MB, "
                  f"{n_in} inputs, {n_out} outputs, {len(groups)} shape group(s) "
                  f"(load+ack {(t1 - t0) * 1000:.2f} ms)")

    # ---- Session API ----

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def close(self) -> None:
        if self._sock is not None:
            try:
                wire.send_frame(self._sock, wire.BYE, b"")
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def __del__(self):  # best-effort
        try:
            self.close()
        except Exception:
            pass

    def get_providers(self):
        return "RemoteAXExecutionProvider"

    def get_inputs(self, shape_group: int = 0):
        return self._inputs[shape_group]

    def get_outputs(self, shape_group: int = 0):
        return self._outputs[shape_group]

    def run(self, output_names, input_feed, run_options=None, shape_group: int = 0):
        if self._sock is None:
            raise RuntimeError("session closed")
        self._validate_input(input_feed)
        self._validate_output(output_names)
        if not (0 <= shape_group < self._shape_count):
            raise ValueError(f"invalid shape_group {shape_group}")

        in_meta = self._group_inputs_meta[shape_group]
        payload = wire.encode_run(shape_group, input_feed, in_meta)
        in_bytes = sum(int(m["size_bytes"]) for m in in_meta)

        t_send0 = time.perf_counter()
        wire.send_frame(self._sock, wire.RUN, payload)
        t_send1 = time.perf_counter()

        mtype, reply = wire.recv_frame(self._sock)
        t_recv = time.perf_counter()

        if mtype == wire.ERROR:
            code, msg = wire.parse_error(reply)
            raise RuntimeError(f"remote RUN failed: code={code} msg={msg}")
        if mtype != wire.RUN_RESULT:
            raise RuntimeError(f"unexpected reply to RUN: type={mtype}")
        status, dev_us, outs = wire.parse_run_result(reply)
        if status != 0:
            raise RuntimeError(f"remote RUN reported status={status}")

        # Reshape outputs and order by `output_names` if provided.
        out_meta = self._group_outputs_meta[shape_group]
        name_to_arr: dict[str, np.ndarray] = {}
        out_bytes = 0
        for (desc, data), meta in zip(outs, out_meta):
            arr = np.frombuffer(data, dtype=desc["dtype"]).reshape(desc["shape"]).copy()
            name_to_arr[desc["name"]] = arr
            out_bytes += len(data)

        if output_names is None:
            ordered = [name_to_arr[m["name"]] for m in out_meta]
        else:
            ordered = [name_to_arr[n] for n in output_names]

        in_ms = (t_send1 - t_send0) * 1000.0
        dev_ms = dev_us / 1000.0
        # Anything between t_send1 and t_recv that isn't device inference is
        # output transfer plus client recv overhead.
        out_ms = max(0.0, (t_recv - t_send1) * 1000.0 - dev_ms)
        total_ms = (t_recv - t_send0) * 1000.0

        self.last_timing = {
            "input_ms":   in_ms,
            "device_ms":  dev_ms,
            "output_ms":  out_ms,
            "total_ms":   total_ms,
            "input_bytes":  int(in_bytes),
            "output_bytes": int(out_bytes),
        }

        if self._verbose:
            print(
                f"[REMOTE {self._endpoint[0]}:{self._endpoint[1]}] "
                f"input(c->d): {in_ms:6.2f} ms ({in_bytes/1e6:5.2f} MB)  "
                f"device(NPU): {dev_ms:6.2f} ms  "
                f"output(d->c): {out_ms:6.2f} ms ({out_bytes/1e6:5.2f} MB)  "
                f"total: {total_ms:6.2f} ms")

        return ordered
