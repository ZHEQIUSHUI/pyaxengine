# Copyright (c) 2026 ax-remote-infer authors. Licensed under BSD-3-Clause.
"""
Wire protocol mirror of ax-remote-infer/device_daemon/include/wire/wire.hpp.

Header (12B):  magic u32 | version u16 | type u16 | payload_size u32
All little-endian on x86_64/aarch64/armv7/riscv64.

Per-tensor descriptor inside payloads:
   u16 name_len, name bytes,
   u8  dtype, u8 ndim,
   u32 shape[ndim], u32 size_bytes [, raw bytes]
"""

from __future__ import annotations

import socket
import struct
from typing import Any, List, Optional, Tuple

import numpy as np

try:
    import ml_dtypes as _mldt  # type: ignore
    _BF16 = np.dtype(_mldt.bfloat16)
except Exception:  # pragma: no cover
    _BF16 = None

WIRE_MAGIC   = 0x52495841  # 'AXIR' little-endian
WIRE_VERSION = 1

# MsgType
HELLO        = 1
HELLO_ACK    = 2
LOAD_MODEL   = 3
MODEL_READY  = 4
RUN          = 5
RUN_RESULT   = 6
BYE          = 7
ERROR        = 99

# ProviderCode
PROVIDER_AUTO       = 0
PROVIDER_AXENGINE   = 1
PROVIDER_AXCLRT     = 2

# DType code -> numpy dtype
_DTYPE_TO_NP = {
    1:  np.dtype(np.uint8),
    2:  np.dtype(np.int8),
    3:  np.dtype(np.uint16),
    4:  np.dtype(np.int16),
    5:  np.dtype(np.uint32),
    6:  np.dtype(np.int32),
    7:  np.dtype(np.uint64),
    8:  np.dtype(np.int64),
    9:  np.dtype(np.float16),
    11: np.dtype(np.float32),
    12: np.dtype(np.float64),
}
if _BF16 is not None:
    _DTYPE_TO_NP[10] = _BF16

_NP_TO_DTYPE = {v: k for k, v in _DTYPE_TO_NP.items()}


def dtype_code(np_dtype) -> int:
    d = np.dtype(np_dtype)
    if d in _NP_TO_DTYPE:
        return _NP_TO_DTYPE[d]
    raise ValueError(f"unsupported numpy dtype for wire protocol: {d}")


def np_dtype(code: int):
    if code in _DTYPE_TO_NP:
        return _DTYPE_TO_NP[code]
    raise ValueError(f"unknown wire dtype code {code}")


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(view[got:], n - got)
        if r == 0:
            raise ConnectionError("peer closed during recv_exact")
        got += r
    return bytes(buf)


def send_frame(sock: socket.socket, msg_type: int, payload: bytes) -> None:
    hdr = struct.pack("<IHHI", WIRE_MAGIC, WIRE_VERSION, msg_type, len(payload))
    if payload:
        sock.sendall(hdr + payload)
    else:
        sock.sendall(hdr)


def recv_frame(sock: socket.socket) -> Tuple[int, bytes]:
    hdr = recv_exact(sock, 12)
    magic, version, mtype, plen = struct.unpack("<IHHI", hdr)
    if magic != WIRE_MAGIC:
        raise ConnectionError(f"bad wire magic 0x{magic:08x}")
    if version != WIRE_VERSION:
        raise ConnectionError(f"unsupported wire version {version}")
    payload = recv_exact(sock, plen) if plen > 0 else b""
    return mtype, payload


def encode_tensor_desc(name: str, np_arr_or_dtype, shape, size_bytes: int) -> bytes:
    if isinstance(np_arr_or_dtype, np.ndarray):
        dt = np_arr_or_dtype.dtype
    else:
        dt = np.dtype(np_arr_or_dtype)
    nbytes = name.encode("utf-8")
    out = bytearray()
    out += struct.pack("<H", len(nbytes))
    out += nbytes
    out += struct.pack("<BB", dtype_code(dt), len(shape))
    out += struct.pack(f"<{len(shape)}I", *[int(s) for s in shape])
    out += struct.pack("<I", int(size_bytes))
    return bytes(out)


def parse_tensor_desc(payload: bytes, offset: int, read_data: bool = False
                      ) -> Tuple[dict, int, Optional[bytes]]:
    end = len(payload)
    if offset + 2 > end:
        raise ValueError("tensor desc truncated (name_len)")
    (name_len,) = struct.unpack_from("<H", payload, offset); offset += 2
    if offset + name_len > end:
        raise ValueError("tensor desc truncated (name)")
    name = payload[offset:offset + name_len].decode("utf-8"); offset += name_len
    if offset + 2 > end:
        raise ValueError("tensor desc truncated (dtype/ndim)")
    dt, ndim = struct.unpack_from("<BB", payload, offset); offset += 2
    if offset + 4 * ndim > end:
        raise ValueError("tensor desc truncated (shape)")
    shape = list(struct.unpack_from(f"<{ndim}I", payload, offset)); offset += 4 * ndim
    if offset + 4 > end:
        raise ValueError("tensor desc truncated (size)")
    (size_bytes,) = struct.unpack_from("<I", payload, offset); offset += 4

    data = None
    if read_data:
        if offset + size_bytes > end:
            raise ValueError("tensor data truncated")
        data = bytes(payload[offset:offset + size_bytes])
        offset += size_bytes

    desc = {"name": name, "dtype": np_dtype(dt), "shape": shape, "size_bytes": size_bytes}
    return desc, offset, data


# ---- Specific messages ----

def encode_hello(requested_provider: int = PROVIDER_AUTO, device_id: int = -1) -> bytes:
    # struct { u32 client_version; u8 requested_provider; i32 device_id; u16 reserved; } packed
    return struct.pack("<IBiH", WIRE_VERSION, requested_provider, device_id, 0)


def decode_hello_ack(payload: bytes) -> dict:
    # struct { u32 server_version; u8 used_provider; u8 vnpu_type; u16 reserved; }
    # followed by u16 chip_type_len, chip_type, u16 engine_version_len, engine_version
    if len(payload) < 8:
        raise ValueError("hello_ack too short")
    server_version, used_provider, vnpu_type, _ = struct.unpack_from("<IBBH", payload, 0)
    off = 8
    (cl,) = struct.unpack_from("<H", payload, off); off += 2
    chip_type = payload[off:off + cl].decode("utf-8", "replace"); off += cl
    (el,) = struct.unpack_from("<H", payload, off); off += 2
    engine_version = payload[off:off + el].decode("utf-8", "replace"); off += el
    return {
        "server_version": server_version,
        "used_provider": used_provider,
        "vnpu_type": vnpu_type,
        "chip_type": chip_type,
        "engine_version": engine_version,
    }


def encode_load_model(model_bytes: bytes) -> bytes:
    return struct.pack("<Q", len(model_bytes)) + model_bytes


def parse_model_ready(payload: bytes) -> List[dict]:
    """
    Returns list[shape_group] where each shape_group is:
        {"inputs": [desc, ...], "outputs": [desc, ...]}
    desc: {name, dtype (np.dtype), shape, size_bytes}
    """
    if len(payload) < 4:
        raise ValueError("model_ready too short")
    off = 0
    (gc,) = struct.unpack_from("<I", payload, off); off += 4
    groups = []
    for _ in range(gc):
        (ic,) = struct.unpack_from("<I", payload, off); off += 4
        inputs = []
        for _ in range(ic):
            desc, off, _data = parse_tensor_desc(payload, off, read_data=False)
            inputs.append(desc)
        (oc,) = struct.unpack_from("<I", payload, off); off += 4
        outputs = []
        for _ in range(oc):
            desc, off, _data = parse_tensor_desc(payload, off, read_data=False)
            outputs.append(desc)
        groups.append({"inputs": inputs, "outputs": outputs})
    return groups


def encode_run(shape_group: int,
               input_feed: dict,
               input_meta: List[dict]) -> bytes:
    """Encode a RUN payload. input_feed: name -> np.ndarray. input_meta describes
    the model inputs for the given shape group (used for ordering + size check)."""
    out = bytearray()
    out += struct.pack("<II", int(shape_group), len(input_meta))
    for meta in input_meta:
        nm = meta["name"]
        if nm not in input_feed:
            raise ValueError(f"missing input '{nm}'")
        arr = input_feed[nm]
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        if arr.dtype != meta["dtype"]:
            raise ValueError(f"input '{nm}' dtype {arr.dtype} != expected {meta['dtype']}")
        if list(arr.shape) != list(meta["shape"]):
            raise ValueError(f"input '{nm}' shape {arr.shape} != expected {meta['shape']}")
        if not (arr.flags.c_contiguous or arr.flags.f_contiguous):
            arr = np.ascontiguousarray(arr)
        nb = arr.nbytes
        if nb != int(meta["size_bytes"]):
            raise ValueError(f"input '{nm}' size {nb} != expected {meta['size_bytes']}")
        out += encode_tensor_desc(nm, arr, meta["shape"], nb)
        out += memoryview(arr).tobytes()
    return bytes(out)


def parse_run_result(payload: bytes) -> Tuple[int, int, List[Tuple[dict, bytes]]]:
    """Returns (status, device_inference_us, [(desc, raw_bytes), ...])."""
    if len(payload) < 20:
        raise ValueError("run_result too short")
    status, dev_us, oc, _ = struct.unpack_from("<iQII", payload, 0)
    off = 20
    outs: List[Tuple[dict, bytes]] = []
    if status == 0:
        for _ in range(oc):
            desc, off, data = parse_tensor_desc(payload, off, read_data=True)
            outs.append((desc, data))
    return status, dev_us, outs


def parse_error(payload: bytes) -> Tuple[int, str]:
    if len(payload) < 8:
        return -1, "short error frame"
    code, msg_len = struct.unpack_from("<iI", payload, 0)
    msg = payload[8:8 + msg_len].decode("utf-8", "replace")
    return code, msg
