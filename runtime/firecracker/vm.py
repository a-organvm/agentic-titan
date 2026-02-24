"""
Firecracker MicroVM Management

Lifecycle management for Firecracker microVMs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from runtime.firecracker.config import FirecrackerConfig, VMState

logger = logging.getLogger("titan.runtime.firecracker.vm")


@dataclass
class ExecutionResult:
    """Result of executing a command in a VM."""

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
class MicroVM:
    """Represents a Firecracker MicroVM instance."""

    vm_id: str = field(default_factory=lambda: str(uuid4())[:8])
    config: FirecrackerConfig = field(default_factory=FirecrackerConfig)
    state: VMState = VMState.CREATED
    process: asyncio.subprocess.Process | None = None
    socket_path: str = ""
    pid: int | None = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    stopped_at: datetime | None = None

    # Metrics
    boot_time_ms: int | None = None
    total_executions: int = 0
    total_errors: int = 0

    # Network info
    tap_device: str | None = None
    vsock_path: str | None = None

    def __post_init__(self) -> None:
        if not self.socket_path:
            self.socket_path = self.config.get_socket_path(self.vm_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vm_id": self.vm_id,
            "state": self.state.value,
            "socket_path": self.socket_path,
            "pid": self.pid,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "boot_time_ms": self.boot_time_ms,
            "total_executions": self.total_executions,
            "total_errors": self.total_errors,
        }


class MicroVMManager:
    """
    Manager for Firecracker MicroVM lifecycle.

    Handles:
    - VM creation and configuration
    - Starting and stopping VMs
    - Command execution within VMs
    - VM pool management for reuse
    """

    def __init__(
        self,
        default_config: FirecrackerConfig | None = None,
        pool_size: int = 0,
    ) -> None:
        self._default_config = default_config or FirecrackerConfig()
        self._pool_size = pool_size
        self._vms: dict[str, MicroVM] = {}
        self._pool: list[MicroVM] = []
        self._lock = asyncio.Lock()

    async def create(
        self,
        config: FirecrackerConfig | None = None,
    ) -> MicroVM:
        """
        Create a new MicroVM.

        Args:
            config: Optional custom configuration

        Returns:
            Created MicroVM (not yet started)
        """
        vm_config = config or self._default_config
        vm = MicroVM(config=vm_config)

        async with self._lock:
            self._vms[vm.vm_id] = vm

        logger.info(f"Created MicroVM {vm.vm_id}")
        return vm

    async def start(self, vm: MicroVM) -> None:
        """
        Start a MicroVM.

        Args:
            vm: MicroVM to start
        """
        if vm.state == VMState.RUNNING:
            logger.warning(f"VM {vm.vm_id} is already running")
            return

        vm.state = VMState.STARTING

        try:
            # Verify Firecracker binary exists
            if not os.path.exists(vm.config.firecracker_path):
                raise FileNotFoundError(
                    f"Firecracker binary not found: {vm.config.firecracker_path}"
                )

            # Create socket directory
            socket_dir = os.path.dirname(vm.socket_path)
            os.makedirs(socket_dir, exist_ok=True)

            # Remove stale socket
            if os.path.exists(vm.socket_path):
                os.remove(vm.socket_path)

            # Build command
            cmd = [
                vm.config.firecracker_path,
                "--api-sock",
                vm.socket_path,
            ]

            if vm.config.log_level:
                cmd.extend(["--level", vm.config.log_level])

            if vm.config.seccomp_level > 0:
                cmd.extend(["--seccomp-level", str(vm.config.seccomp_level)])

            # Start Firecracker process
            start_time = datetime.now()
            vm.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            vm.pid = vm.process.pid

            # Wait for socket to be available
            await self._wait_for_socket(vm.socket_path, timeout=5.0)

            # Configure the VM via API
            await self._configure_vm(vm)

            # Start the VM instance
            await self._api_put(vm, "/actions", {"action_type": "InstanceStart"})

            # Wait for boot
            await self._wait_for_boot(vm)

            vm.state = VMState.RUNNING
            vm.started_at = datetime.now()
            vm.boot_time_ms = int((vm.started_at - start_time).total_seconds() * 1000)

            logger.info(f"Started MicroVM {vm.vm_id} in {vm.boot_time_ms}ms")

        except Exception as e:
            vm.state = VMState.ERROR
            logger.error(f"Failed to start VM {vm.vm_id}: {e}")
            await self._cleanup_vm(vm)
            raise

    async def stop(self, vm: MicroVM, force: bool = False) -> None:
        """
        Stop a MicroVM.

        Args:
            vm: MicroVM to stop
            force: If True, force kill without graceful shutdown
        """
        if vm.state not in (VMState.RUNNING, VMState.PAUSED, VMState.STARTING):
            return

        vm.state = VMState.STOPPING

        try:
            if not force:
                # Try graceful shutdown via API
                try:
                    await self._api_put(vm, "/actions", {"action_type": "SendCtrlAltDel"})
                    await asyncio.sleep(1.0)
                except Exception:
                    pass

            # Kill process if still running
            if vm.process and vm.process.returncode is None:
                vm.process.terminate()
                try:
                    await asyncio.wait_for(vm.process.wait(), timeout=5.0)
                except TimeoutError:
                    vm.process.kill()
                    await vm.process.wait()

        except Exception as e:
            logger.warning(f"Error stopping VM {vm.vm_id}: {e}")

        await self._cleanup_vm(vm)

        vm.state = VMState.STOPPED
        vm.stopped_at = datetime.now()

        async with self._lock:
            self._vms.pop(vm.vm_id, None)

        logger.info(f"Stopped MicroVM {vm.vm_id}")

    async def execute(
        self,
        vm: MicroVM,
        command: str,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """
        Execute a command in a MicroVM.

        This uses the guest agent (via VSOCK or serial) for communication.

        Args:
            vm: Target MicroVM
            command: Command to execute
            timeout: Optional timeout override

        Returns:
            ExecutionResult with output and exit code
        """
        if vm.state != VMState.RUNNING:
            return ExecutionResult(
                exit_code=-1,
                error=f"VM is not running (state: {vm.state.value})",
            )

        timeout = timeout or vm.config.timeout_seconds
        start_time = datetime.now()
        vm.total_executions += 1

        try:
            # Import guest agent here to avoid circular imports
            from runtime.firecracker.guest_agent import GuestAgentProtocol

            agent = GuestAgentProtocol()
            cmd_result = await agent.execute_command_via_api(vm, command, timeout)

            result = ExecutionResult(
                exit_code=cmd_result.exit_code,
                stdout=cmd_result.stdout,
                stderr=cmd_result.stderr,
                timed_out=cmd_result.timed_out,
                error=cmd_result.error,
            )
            result.duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            return result

        except TimeoutError:
            vm.total_errors += 1
            return ExecutionResult(
                exit_code=-1,
                timed_out=True,
                duration_ms=int(timeout * 1000),
                error=f"Command timed out after {timeout}s",
            )

        except Exception as e:
            vm.total_errors += 1
            return ExecutionResult(
                exit_code=-1,
                error=str(e),
                duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
            )

    async def copy_to_vm(
        self,
        vm: MicroVM,
        local_path: str,
        remote_path: str,
    ) -> bool:
        """
        Copy a file to the VM.

        Args:
            vm: Target MicroVM
            local_path: Local file path
            remote_path: Path inside VM

        Returns:
            True if successful
        """
        if vm.state != VMState.RUNNING:
            return False

        try:
            with open(local_path, "rb") as f:
                content = f.read()

            from runtime.firecracker.guest_agent import GuestAgentProtocol

            agent = GuestAgentProtocol()
            return await agent.write_file_via_api(vm, remote_path, content)

        except Exception as e:
            logger.error(f"Failed to copy to VM {vm.vm_id}: {e}")
            return False

    async def copy_from_vm(
        self,
        vm: MicroVM,
        remote_path: str,
    ) -> bytes | None:
        """
        Copy a file from the VM.

        Args:
            vm: Source MicroVM
            remote_path: Path inside VM

        Returns:
            File contents or None if failed
        """
        if vm.state != VMState.RUNNING:
            return None

        try:
            from runtime.firecracker.guest_agent import GuestAgentProtocol

            agent = GuestAgentProtocol()
            return await agent.read_file_via_api(vm, remote_path)

        except Exception as e:
            logger.error(f"Failed to copy from VM {vm.vm_id}: {e}")
            return None

    async def get_or_create_from_pool(
        self,
        config: FirecrackerConfig | None = None,
    ) -> MicroVM:
        """
        Get a VM from pool or create new one.

        Args:
            config: Configuration for new VM if needed

        Returns:
            Ready-to-use MicroVM
        """
        async with self._lock:
            if self._pool:
                vm = self._pool.pop(0)
                logger.debug(f"Got VM {vm.vm_id} from pool")
                return vm

        # Create and start new VM
        vm = await self.create(config)
        await self.start(vm)
        return vm

    async def return_to_pool(self, vm: MicroVM) -> None:
        """
        Return a VM to the pool for reuse.

        Args:
            vm: MicroVM to return
        """
        if vm.state != VMState.RUNNING:
            await self.stop(vm)
            return

        should_stop = False
        async with self._lock:
            if len(self._pool) < self._pool_size:
                self._pool.append(vm)
                logger.debug(f"Returned VM {vm.vm_id} to pool")
            else:
                # Pool full, will stop the VM after releasing lock
                should_stop = True

        if should_stop:
            await self.stop(vm)

    async def _configure_vm(self, vm: MicroVM) -> None:
        """Configure VM via Firecracker API."""
        # Machine configuration
        await self._api_put(vm, "/machine-config", vm.config.to_machine_config())

        # Boot source
        await self._api_put(vm, "/boot-source", vm.config.to_boot_source())

        # Root drive
        await self._api_put(
            vm,
            "/drives/rootfs",
            vm.config.to_drive_config("rootfs"),
        )

        # Network (if enabled)
        network_config = vm.config.to_network_config()
        if network_config:
            await self._api_put(vm, "/network-interfaces/eth0", network_config)

        # VSOCK (if enabled)
        vsock_config = vm.config.to_vsock_config()
        if vsock_config:
            await self._api_put(vm, "/vsock", vsock_config)
            vm.vsock_path = vsock_config.get("uds_path")

        # Logger (if configured)
        logger_config = vm.config.to_logger_config()
        if logger_config:
            await self._api_put(vm, "/logger", logger_config)

    async def _api_put(
        self,
        vm: MicroVM,
        endpoint: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Make PUT request to Firecracker API."""
        try:
            import aiohttp

            async with aiohttp.UnixConnector(path=vm.socket_path) as connector:
                async with aiohttp.ClientSession(connector=connector) as session:
                    url = f"http://localhost{endpoint}"
                    async with session.put(url, json=data) as response:
                        if response.status >= 400:
                            text = await response.text()
                            raise RuntimeError(f"API error {response.status}: {text}")
                        if response.content_length and response.content_length > 0:
                            result: dict[str, Any] = await response.json()
                            return result
                        return {}

        except ImportError:
            # Fallback to curl
            return await self._api_put_curl(vm, endpoint, data)

    async def _api_put_curl(
        self,
        vm: MicroVM,
        endpoint: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Make PUT request using curl (fallback)."""
        cmd = [
            "curl",
            "--unix-socket",
            vm.socket_path,
            "-X",
            "PUT",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(data),
            f"http://localhost{endpoint}",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"curl failed: {stderr.decode()}")

        if stdout:
            result: dict[str, Any] = json.loads(stdout.decode())
            return result
        return {}

    async def _api_get(
        self,
        vm: MicroVM,
        endpoint: str,
    ) -> dict[str, Any]:
        """Make GET request to Firecracker API."""
        try:
            import aiohttp

            async with aiohttp.UnixConnector(path=vm.socket_path) as connector:
                async with aiohttp.ClientSession(connector=connector) as session:
                    url = f"http://localhost{endpoint}"
                    async with session.get(url) as response:
                        result: dict[str, Any] = await response.json()
                        return result

        except ImportError:
            # Fallback to curl
            cmd = [
                "curl",
                "-s",
                "--unix-socket",
                vm.socket_path,
                f"http://localhost{endpoint}",
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            return json.loads(stdout.decode()) if stdout else {}

    async def _wait_for_socket(self, socket_path: str, timeout: float) -> None:
        """Wait for socket to become available."""
        start = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start < timeout:
            if os.path.exists(socket_path):
                return
            await asyncio.sleep(0.1)

        raise TimeoutError(f"Socket {socket_path} not available after {timeout}s")

    async def _wait_for_boot(self, vm: MicroVM) -> None:
        """Wait for VM to finish booting."""
        timeout = vm.config.boot_timeout_seconds
        start = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start < timeout:
            try:
                info = await self._api_get(vm, "/")
                if info.get("state") == "Running":
                    return
            except Exception:
                pass
            await asyncio.sleep(0.2)

        # Don't fail - VM might still be usable
        logger.warning(f"VM {vm.vm_id} boot verification timed out")

    async def _cleanup_vm(self, vm: MicroVM) -> None:
        """Clean up VM resources."""
        # Remove socket
        if os.path.exists(vm.socket_path):
            try:
                os.remove(vm.socket_path)
            except Exception:
                pass

        # Remove VSOCK path
        if vm.vsock_path and os.path.exists(vm.vsock_path):
            try:
                os.remove(vm.vsock_path)
            except Exception:
                pass

        # Clean up TAP device
        if vm.tap_device:
            try:
                from runtime.firecracker.network import get_network_manager

                network = get_network_manager()
                await network.cleanup_tap_device(vm.tap_device)
            except Exception:
                pass

    async def list_vms(self) -> list[MicroVM]:
        """List all managed VMs."""
        async with self._lock:
            return list(self._vms.values())

    async def shutdown(self) -> None:
        """Shutdown all VMs and the manager."""
        async with self._lock:
            # Collect all VMs and clear pool
            vms = list(self._vms.values()) + self._pool
            self._pool.clear()

        for vm in vms:
            await self.stop(vm, force=True)

        logger.info("MicroVM manager shutdown complete")


# Singleton instance
_manager: MicroVMManager | None = None


def get_vm_manager(
    config: FirecrackerConfig | None = None,
    pool_size: int = 0,
) -> MicroVMManager:
    """Get the default MicroVM manager."""
    global _manager
    if _manager is None:
        _manager = MicroVMManager(config, pool_size)
    return _manager
