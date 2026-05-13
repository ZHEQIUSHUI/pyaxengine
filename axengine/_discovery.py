# Copyright (c) 2026 ax-remote-infer authors. Licensed under BSD-3-Clause.
"""
LAN discovery for ax_remote_infer devices.

Listens on UDP for broadcast packets emitted by the device-side daemon.
Optionally also probes mDNS if `zeroconf` is installed.

A discovered device looks like:

    DeviceInfo(
        ip="192.168.1.42",
        tcp_port=18500,
        chip_type="AX650N",
        backend="AxEngineExecutionProvider",
        hostname="ax650-dev01",
        ts=1715587200,
    )
"""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

DEFAULT_BROADCAST_PORT = 9988
DEFAULT_TCP_PORT = 18500
DEFAULT_SERVICE_NAME = "ax_remote_infer"


@dataclass(frozen=True)
class DeviceInfo:
    ip: str
    tcp_port: int = DEFAULT_TCP_PORT
    chip_type: str = ""
    backend: str = ""
    hostname: str = ""
    service_version: int = 1
    ts: int = 0

    def endpoint(self) -> tuple:
        return (self.ip, self.tcp_port)


def _udp_listen(timeout: float, port: int) -> List[DeviceInfo]:
    seen: dict[tuple, DeviceInfo] = {}
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass
        sock.bind(("", port))
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            sock.settimeout(remaining)
            try:
                data, addr = sock.recvfrom(8192)
            except socket.timeout:
                break
            except OSError:
                break
            try:
                obj = json.loads(data.decode("utf-8", "replace"))
            except Exception:
                continue
            if obj.get("service") != DEFAULT_SERVICE_NAME:
                continue
            ip = addr[0]
            tcp_port = int(obj.get("tcp_port", DEFAULT_TCP_PORT))
            key = (ip, tcp_port)
            seen[key] = DeviceInfo(
                ip=ip,
                tcp_port=tcp_port,
                chip_type=str(obj.get("chip_type", "")),
                backend=str(obj.get("backend", "")),
                hostname=str(obj.get("hostname", "")),
                service_version=int(obj.get("service_version", 1)),
                ts=int(obj.get("ts", 0)),
            )
    finally:
        sock.close()
    return list(seen.values())


def _mdns_browse(timeout: float) -> List[DeviceInfo]:
    """Optional zeroconf browse; returns [] if zeroconf is not installed."""
    try:
        from zeroconf import ServiceBrowser, Zeroconf  # type: ignore
    except Exception:
        return []

    found: dict[tuple, DeviceInfo] = {}

    class _Listener:
        def add_service(self, zc, type_, name):
            info = zc.get_service_info(type_, name, timeout=int(timeout * 1000))
            if not info:
                return
            ip = socket.inet_ntoa(info.addresses[0]) if info.addresses else ""
            if not ip:
                return
            props = {(k.decode() if isinstance(k, bytes) else k):
                     (v.decode() if isinstance(v, bytes) else v)
                     for k, v in (info.properties or {}).items()}
            tcp_port = int(props.get("tcp_port", info.port or DEFAULT_TCP_PORT))
            key = (ip, tcp_port)
            found[key] = DeviceInfo(
                ip=ip,
                tcp_port=tcp_port,
                chip_type=str(props.get("chip_type", "")),
                backend=str(props.get("backend", "")),
                hostname=str(props.get("hostname", "")),
                service_version=int(props.get("service_version", 1)),
                ts=int(time.time()),
            )

        def remove_service(self, *a, **kw):  # noqa: D401
            pass

        def update_service(self, *a, **kw):
            pass

    zc = Zeroconf()
    try:
        ServiceBrowser(zc, "_axinfer._tcp.local.", _Listener())
        time.sleep(timeout)
    finally:
        zc.close()
    return list(found.values())


def discover_devices(
    timeout: float = 2.0,
    methods: Iterable[str] = ("udp",),
    broadcast_port: int = DEFAULT_BROADCAST_PORT,
) -> List[DeviceInfo]:
    """Return all ax_remote_infer devices visible on the LAN.

    Parameters
    ----------
    timeout : float
        Total listen window (seconds). 2s typically catches ≥1 broadcast cycle.
    methods : iterable of str
        Any of {"udp", "mdns"}.  Default is UDP only.
    broadcast_port : int
        UDP port the daemon broadcasts on (default 9988).
    """
    out: dict[tuple, DeviceInfo] = {}
    methods = set(methods)
    if "udp" in methods:
        for d in _udp_listen(timeout=timeout, port=broadcast_port):
            out[d.endpoint()] = d
    if "mdns" in methods:
        for d in _mdns_browse(timeout=timeout):
            out[d.endpoint()] = d
    return list(out.values())
