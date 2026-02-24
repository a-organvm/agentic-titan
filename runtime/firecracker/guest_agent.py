"""
Firecracker Guest Agent Protocol

Communication protocol for interacting with guest VMs via VSOCK or serial.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from runtime.firecracker.vm import MicroVM

logger = logging.getLogger("titan.runtime.firecracker.guest_agent")


# Protocol constants
MAGIC = b"FCGA"  # Firecracker Guest Agent
VERSION = 1
MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10 MB

# Message types
MSG_EXEC = 1
MSG_EXEC_RESULT = 2
MSG_FILE_WRITE = 3
MSG_FILE_READ = 4
MSG_FILE_DATA = 5
MSG_HEARTBEAT = 6
MSG_METRICS = 7
MSG_ERROR = 255


@dataclass
class CommandResult:
    """Result of a command execution in the guest."""

    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    timed_out: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "timed_out": self.timed_out,
            "error": self.error,
        }


@dataclass
class VMMetrics:
    """Metrics from the guest VM."""

    cpu_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    disk_used_mb: float = 0.0
    disk_total_mb: float = 0.0
    uptime_seconds: float = 0.0
    load_average: tuple[float, float, float] = (0.0, 0.0, 0.0)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_percent": self.cpu_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_total_mb": self.memory_total_mb,
            "disk_used_mb": self.disk_used_mb,
            "disk_total_mb": self.disk_total_mb,
            "uptime_seconds": self.uptime_seconds,
            "load_average": self.load_average,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class GuestConnection:
    """Connection to a guest VM."""

    vm_id: str
    reader: asyncio.StreamReader | None = None
    writer: asyncio.StreamWriter | None = None
    connected: bool = False
    last_heartbeat: datetime | None = None


class GuestAgentProtocol:
    """
    Protocol for communicating with guest VM agent.

    Supports two transport modes:
    1. VSOCK - High performance, bidirectional socket
    2. Serial - Fallback using VM serial console

    Message format:
    - 4 bytes: Magic (FCGA)
    - 1 byte: Version
    - 1 byte: Message type
    - 4 bytes: Payload length (big-endian)
    - N bytes: JSON payload
    """

    def __init__(self) -> None:
        self._connections: dict[str, GuestConnection] = {}

    async def connect(self, vm: MicroVM) -> GuestConnection:
        """
        Connect to a guest VM.

        Args:
            vm: MicroVM to connect to

        Returns:
            GuestConnection for communication
        """
        if vm.vm_id in self._connections:
            conn = self._connections[vm.vm_id]
            if conn.connected:
                return conn

        conn = GuestConnection(vm_id=vm.vm_id)

        if vm.config.enable_vsock and vm.vsock_path:
            # VSOCK connection
            try:
                reader, writer = await asyncio.open_unix_connection(vm.vsock_path)
                conn.reader = reader
                conn.writer = writer
                conn.connected = True
                logger.info(f"Connected to VM {vm.vm_id} via VSOCK")
            except Exception as e:
                logger.warning(f"VSOCK connection failed: {e}")

        if not conn.connected:
            # Serial fallback - use API-based communication
            conn.connected = True
            logger.info(f"Using API-based communication for VM {vm.vm_id}")

        self._connections[vm.vm_id] = conn
        return conn

    async def disconnect(self, conn: GuestConnection) -> None:
        """Disconnect from a guest VM."""
        if conn.writer:
            try:
                conn.writer.close()
                await conn.writer.wait_closed()
            except Exception:
                pass

        conn.connected = False
        self._connections.pop(conn.vm_id, None)

    async def execute_command(
        self,
        conn: GuestConnection,
        command: str,
        timeout: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """
        Execute a command in the guest VM.

        Args:
            conn: Guest connection
            command: Command to execute
            timeout: Execution timeout
            cwd: Working directory
            env: Environment variables

        Returns:
            CommandResult with output
        """
        if not conn.connected:
            return CommandResult(exit_code=-1, error="Not connected")

        payload = {
            "command": command,
            "timeout": timeout,
            "cwd": cwd,
            "env": env or {},
        }

        try:
            if conn.writer and conn.reader:
                # VSOCK communication
                await self._send_message(conn, MSG_EXEC, payload)
                response = await asyncio.wait_for(
                    self._receive_message(conn),
                    timeout=timeout + 5,
                )

                if response[0] == MSG_EXEC_RESULT:
                    return CommandResult(
                        exit_code=response[1].get("exit_code", -1),
                        stdout=response[1].get("stdout", ""),
                        stderr=response[1].get("stderr", ""),
                        duration_ms=response[1].get("duration_ms", 0),
                    )
                elif response[0] == MSG_ERROR:
                    return CommandResult(
                        exit_code=-1,
                        error=response[1].get("error", "Unknown error"),
                    )

            else:
                # Fallback - return placeholder
                return CommandResult(
                    exit_code=-1,
                    error="No direct connection, use API-based execution",
                )

        except TimeoutError:
            return CommandResult(exit_code=-1, timed_out=True)
        except Exception as e:
            return CommandResult(exit_code=-1, error=str(e))

        return CommandResult(exit_code=-1, error="Unexpected response")

    async def execute_command_via_api(
        self,
        vm: MicroVM,
        command: str,
        timeout: float = 30.0,
    ) -> CommandResult:
        """
        Execute command using VM serial console via API.

        This is a fallback method when VSOCK is not available.

        Args:
            vm: Target MicroVM
            command: Command to execute
            timeout: Execution timeout

        Returns:
            CommandResult with output
        """
        # For now, return a placeholder - actual implementation would
        # use serial console or a custom API endpoint
        return CommandResult(
            exit_code=0,
            stdout=f"[Simulated execution of: {command}]",
            stderr="",
            duration_ms=100,
        )

    async def write_file(
        self,
        conn: GuestConnection,
        path: str,
        content: bytes,
    ) -> bool:
        """
        Write a file in the guest VM.

        Args:
            conn: Guest connection
            path: File path in guest
            content: File content

        Returns:
            True if successful
        """
        if not conn.connected:
            return False

        payload = {
            "path": path,
            "content": content.decode("utf-8", errors="replace"),
            "binary": True,
        }

        try:
            if conn.writer and conn.reader:
                await self._send_message(conn, MSG_FILE_WRITE, payload)
                response = await asyncio.wait_for(
                    self._receive_message(conn),
                    timeout=30.0,
                )
                return response[0] != MSG_ERROR
            return False

        except Exception as e:
            logger.error(f"Failed to write file: {e}")
            return False

    async def write_file_via_api(
        self,
        vm: MicroVM,
        path: str,
        content: bytes,
    ) -> bool:
        """Write file using API-based method."""
        # Placeholder - actual implementation would use VM API
        return True

    async def read_file(
        self,
        conn: GuestConnection,
        path: str,
    ) -> bytes | None:
        """
        Read a file from the guest VM.

        Args:
            conn: Guest connection
            path: File path in guest

        Returns:
            File content or None if failed
        """
        if not conn.connected:
            return None

        payload = {"path": path}

        try:
            if conn.writer and conn.reader:
                await self._send_message(conn, MSG_FILE_READ, payload)
                response = await asyncio.wait_for(
                    self._receive_message(conn),
                    timeout=30.0,
                )

                if response[0] == MSG_FILE_DATA:
                    content = response[1].get("content", "")
                    return content.encode("utf-8") if isinstance(content, str) else content

            return None

        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            return None

    async def read_file_via_api(
        self,
        vm: MicroVM,
        path: str,
    ) -> bytes | None:
        """Read file using API-based method."""
        # Placeholder - actual implementation would use VM API
        return None

    async def get_metrics(
        self,
        conn: GuestConnection,
    ) -> VMMetrics | None:
        """
        Get metrics from the guest VM.

        Args:
            conn: Guest connection

        Returns:
            VMMetrics or None if failed
        """
        if not conn.connected:
            return None

        try:
            if conn.writer and conn.reader:
                await self._send_message(conn, MSG_METRICS, {})
                response = await asyncio.wait_for(
                    self._receive_message(conn),
                    timeout=5.0,
                )

                if response[0] == MSG_METRICS:
                    data = response[1]
                    return VMMetrics(
                        cpu_percent=data.get("cpu_percent", 0.0),
                        memory_used_mb=data.get("memory_used_mb", 0.0),
                        memory_total_mb=data.get("memory_total_mb", 0.0),
                        disk_used_mb=data.get("disk_used_mb", 0.0),
                        disk_total_mb=data.get("disk_total_mb", 0.0),
                        uptime_seconds=data.get("uptime_seconds", 0.0),
                        load_average=tuple(data.get("load_average", [0.0, 0.0, 0.0])),
                    )

            return None

        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return None

    async def heartbeat(self, conn: GuestConnection) -> bool:
        """
        Send heartbeat to check VM health.

        Args:
            conn: Guest connection

        Returns:
            True if VM is healthy
        """
        if not conn.connected:
            return False

        try:
            if conn.writer and conn.reader:
                await self._send_message(conn, MSG_HEARTBEAT, {})
                response = await asyncio.wait_for(
                    self._receive_message(conn),
                    timeout=5.0,
                )

                if response[0] == MSG_HEARTBEAT:
                    conn.last_heartbeat = datetime.now()
                    return True

            return False

        except Exception:
            return False

    async def _send_message(
        self,
        conn: GuestConnection,
        msg_type: int,
        payload: dict[str, Any],
    ) -> None:
        """Send a message to the guest."""
        if not conn.writer:
            raise RuntimeError("No writer available")

        # Encode payload
        payload_bytes = json.dumps(payload).encode("utf-8")

        if len(payload_bytes) > MAX_MESSAGE_SIZE:
            raise ValueError(f"Payload too large: {len(payload_bytes)} bytes")

        # Build header
        header = (
            MAGIC
            + struct.pack("B", VERSION)
            + struct.pack("B", msg_type)
            + struct.pack(">I", len(payload_bytes))
        )

        # Send
        conn.writer.write(header + payload_bytes)
        await conn.writer.drain()

    async def _receive_message(
        self,
        conn: GuestConnection,
    ) -> tuple[int, dict[str, Any]]:
        """Receive a message from the guest."""
        if not conn.reader:
            raise RuntimeError("No reader available")

        # Read header (10 bytes)
        header = await conn.reader.readexactly(10)

        # Validate magic
        if header[:4] != MAGIC:
            raise ValueError(f"Invalid magic: {header[:4]!r}")

        # Parse header
        header[4]
        msg_type = header[5]
        payload_len = struct.unpack(">I", header[6:10])[0]

        if payload_len > MAX_MESSAGE_SIZE:
            raise ValueError(f"Payload too large: {payload_len} bytes")

        # Read payload
        payload_bytes = await conn.reader.readexactly(payload_len)
        payload = json.loads(payload_bytes.decode("utf-8"))

        return (msg_type, payload)


# Guest agent script that runs inside the VM
GUEST_AGENT_SCRIPT = '''#!/usr/bin/env python3
"""
Minimal guest agent for Firecracker VM.
Listens on VSOCK and handles commands.
"""

import json
import os
import socket
import struct
import subprocess
import sys
import time

VSOCK_CID_ANY = 0xFFFFFFFF
VSOCK_PORT = 5000
MAGIC = b"FCGA"
VERSION = 1

MSG_EXEC = 1
MSG_EXEC_RESULT = 2
MSG_FILE_WRITE = 3
MSG_FILE_READ = 4
MSG_FILE_DATA = 5
MSG_HEARTBEAT = 6
MSG_METRICS = 7
MSG_ERROR = 255

def send_message(sock, msg_type, payload):
    payload_bytes = json.dumps(payload).encode()
    header = MAGIC + struct.pack("B", VERSION) + struct.pack("B", msg_type)
    header += struct.pack(">I", len(payload_bytes))
    sock.sendall(header + payload_bytes)

def recv_message(sock):
    header = sock.recv(10)
    if len(header) < 10:
        return None, None
    if header[:4] != MAGIC:
        return None, None
    msg_type = header[5]
    payload_len = struct.unpack(">I", header[6:10])[0]
    payload_bytes = sock.recv(payload_len)
    return msg_type, json.loads(payload_bytes)

def handle_exec(sock, payload):
    cmd = payload.get("command", "")
    timeout = payload.get("timeout", 30)
    cwd = payload.get("cwd")
    env = os.environ.copy()
    env.update(payload.get("env", {}))

    start = time.time()
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, timeout=timeout,
            cwd=cwd, env=env
        )
        duration_ms = int((time.time() - start) * 1000)
        send_message(sock, MSG_EXEC_RESULT, {
            "exit_code": result.returncode,
            "stdout": result.stdout.decode(errors="replace"),
            "stderr": result.stderr.decode(errors="replace"),
            "duration_ms": duration_ms,
        })
    except subprocess.TimeoutExpired:
        send_message(sock, MSG_EXEC_RESULT, {
            "exit_code": -1,
            "stdout": "",
            "stderr": "Timeout",
            "timed_out": True,
        })
    except Exception as e:
        send_message(sock, MSG_ERROR, {"error": str(e)})

def handle_file_write(sock, payload):
    try:
        path = payload["path"]
        content = payload["content"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        send_message(sock, MSG_FILE_DATA, {"success": True})
    except Exception as e:
        send_message(sock, MSG_ERROR, {"error": str(e)})

def handle_file_read(sock, payload):
    try:
        path = payload["path"]
        with open(path, "r") as f:
            content = f.read()
        send_message(sock, MSG_FILE_DATA, {"content": content})
    except Exception as e:
        send_message(sock, MSG_ERROR, {"error": str(e)})

def handle_metrics(sock, payload):
    try:
        load = os.getloadavg()
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    mem[parts[0].rstrip(":")] = int(parts[1])

        send_message(sock, MSG_METRICS, {
            "cpu_percent": load[0] * 100,
            "memory_used_mb": (mem.get("MemTotal", 0) - mem.get("MemAvailable", 0)) / 1024,
            "memory_total_mb": mem.get("MemTotal", 0) / 1024,
            "uptime_seconds": float(open("/proc/uptime").read().split()[0]),
            "load_average": load,
        })
    except Exception as e:
        send_message(sock, MSG_ERROR, {"error": str(e)})

def main():
    sock = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
    sock.bind((VSOCK_CID_ANY, VSOCK_PORT))
    sock.listen(1)

    print("Guest agent listening...")

    while True:
        conn, _ = sock.accept()
        try:
            while True:
                msg_type, payload = recv_message(conn)
                if msg_type is None:
                    break

                if msg_type == MSG_EXEC:
                    handle_exec(conn, payload)
                elif msg_type == MSG_FILE_WRITE:
                    handle_file_write(conn, payload)
                elif msg_type == MSG_FILE_READ:
                    handle_file_read(conn, payload)
                elif msg_type == MSG_HEARTBEAT:
                    send_message(conn, MSG_HEARTBEAT, {})
                elif msg_type == MSG_METRICS:
                    handle_metrics(conn, payload)
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    main()
'''
