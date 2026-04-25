"""UDP telemetry sender for real-time SalahSense data streaming."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import socket
import struct


class UdpTelemetrySender:
    """Send JSON telemetry packets over UDP."""

    def __init__(self, interface_name: str, port: int, enabled: bool = True) -> None:
        self.interface_name = interface_name
        self.port = port
        self.enabled = enabled
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.source_ip = _get_iface_ipv4(interface_name)
        self.broadcast_ip = _get_iface_broadcast_ipv4(interface_name)

    def send(self, payload: dict) -> None:
        """Send one telemetry payload as UTF-8 JSON."""
        if not self.enabled:
            return

        packet = {
            "sent_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_interface": self.interface_name,
            "source_ip": self.source_ip,
            **payload,
        }
        message = json.dumps(packet, ensure_ascii=True).encode("utf-8")
        self._socket.sendto(message, (self.broadcast_ip, self.port))

    def close(self) -> None:
        self._socket.close()


def _get_iface_ipv4(interface_name: str) -> str:
    return _ioctl_ipv4(interface_name, request=0x8915)  # SIOCGIFADDR


def _get_iface_broadcast_ipv4(interface_name: str) -> str:
    return _ioctl_ipv4(interface_name, request=0x8919)  # SIOCGIFBRDADDR


def _ioctl_ipv4(interface_name: str, request: int) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ifreq = struct.pack("256s", interface_name[:15].encode("utf-8"))
    try:
        res = fcntl.ioctl(sock.fileno(), request, ifreq)
    except OSError as exc:
        raise RuntimeError(f"Failed to query interface '{interface_name}': {exc}") from exc
    finally:
        sock.close()
    return socket.inet_ntoa(res[20:24])
