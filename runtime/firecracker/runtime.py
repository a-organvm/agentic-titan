"""
Firecracker Runtime Implementation

Runtime implementation for executing agent tasks in Firecracker microVMs.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

from runtime.base import (
    AgentProcess,
    ProcessState,
    Runtime,
    RuntimeConfig,
    RuntimeType,
)
from runtime.firecracker.config import FirecrackerConfig, VMState
from runtime.firecracker.vm import ExecutionResult, MicroVM, MicroVMManager

logger = logging.getLogger("titan.runtime.firecracker")


class FirecrackerRuntime(Runtime):
    """
    Runtime implementation using Firecracker microVMs.

    Provides strong isolation through hardware virtualization with
    sub-second boot times and minimal overhead.

    Features:
    - VM pooling for fast reuse
    - Automatic VM lifecycle management
    - Code and file transfer to/from VMs
    - Timeout and resource enforcement
    """

    type = RuntimeType.LOCAL  # Registered as LOCAL until enum is updated

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        firecracker_config: FirecrackerConfig | None = None,
        pool_size: int = 2,
    ) -> None:
        super().__init__(config)
        self._fc_config = firecracker_config or FirecrackerConfig()
        self._pool_size = pool_size
        self._vm_manager: MicroVMManager | None = None
        self._active_vms: dict[str, MicroVM] = {}

    async def initialize(self) -> None:
        """Initialize the Firecracker runtime."""
        if self._initialized:
            return

        # Check platform availability
        from runtime.firecracker import FIRECRACKER_AVAILABLE, KVM_AVAILABLE

        if not FIRECRACKER_AVAILABLE:
            raise RuntimeError("Firecracker requires Linux")

        if not KVM_AVAILABLE:
            raise RuntimeError("KVM not available - check /dev/kvm permissions")

        # Check Firecracker binary
        if not os.path.exists(self._fc_config.firecracker_path):
            raise FileNotFoundError(
                f"Firecracker binary not found: {self._fc_config.firecracker_path}"
            )

        # Check kernel and rootfs
        if not os.path.exists(self._fc_config.kernel_path):
            logger.warning(f"Kernel not found: {self._fc_config.kernel_path}")

        if not os.path.exists(self._fc_config.rootfs_path):
            logger.warning(f"Rootfs not found: {self._fc_config.rootfs_path}")

        # Initialize VM manager
        self._vm_manager = MicroVMManager(
            default_config=self._fc_config,
            pool_size=self._pool_size,
        )

        self._initialized = True
        logger.info("Firecracker runtime initialized")

    async def shutdown(self) -> None:
        """Shutdown the Firecracker runtime."""
        if self._vm_manager:
            await self._vm_manager.shutdown()

        self._initialized = False
        logger.info("Firecracker runtime shutdown")

    async def spawn(
        self,
        agent_id: str,
        agent_spec: dict[str, Any],
        prompt: str | None = None,
    ) -> AgentProcess:
        """
        Spawn an agent in a Firecracker microVM.

        Args:
            agent_id: Unique agent identifier
            agent_spec: Agent specification
            prompt: Optional initial prompt

        Returns:
            AgentProcess representing the VM
        """
        if not self._initialized:
            await self.initialize()

        process = AgentProcess(
            agent_id=agent_id,
            runtime_type=self.type,
            state=ProcessState.STARTING,
            metadata={"agent_spec": agent_spec},
        )
        self._register_process(process)

        assert self._vm_manager is not None, "Runtime not initialized"

        try:
            # Get or create VM
            vm = await self._vm_manager.get_or_create_from_pool(self._fc_config)

            self._active_vms[process.process_id] = vm
            process.container_id = vm.vm_id
            process.metadata["vm_id"] = vm.vm_id

            # Prepare agent code in VM
            await self._prepare_agent(vm, agent_spec, prompt)

            process.mark_started()
            logger.info(f"Spawned agent {agent_id} in VM {vm.vm_id}")

        except Exception as e:
            process.mark_failed(str(e))
            raise

        return process

    async def stop(self, process_id: str, force: bool = False) -> bool:
        """
        Stop an agent process.

        Args:
            process_id: Process ID
            force: Force stop without graceful shutdown

        Returns:
            True if stopped successfully
        """
        process = self.get_process(process_id)
        if not process:
            return False

        vm = self._active_vms.pop(process_id, None)

        if vm and self._vm_manager:
            if force:
                await self._vm_manager.stop(vm, force=True)
            else:
                # Try to return to pool for reuse
                await self._vm_manager.return_to_pool(vm)

        process.state = ProcessState.CANCELLED if not force else ProcessState.FAILED
        process.completed_at = datetime.now()

        return True

    async def get_status(self, process_id: str) -> AgentProcess | None:
        """Get process status."""
        process = self.get_process(process_id)
        if not process:
            return None

        # Update from VM state if available
        vm = self._active_vms.get(process_id)
        if vm:
            if vm.state == VMState.RUNNING:
                process.state = ProcessState.RUNNING
            elif vm.state == VMState.ERROR:
                process.state = ProcessState.FAILED
            elif vm.state == VMState.STOPPED:
                process.state = ProcessState.COMPLETED

        return process

    async def get_logs(self, process_id: str, tail: int = 100) -> list[str]:
        """Get logs from a process."""
        vm = self._active_vms.get(process_id)
        if not vm:
            return []

        # Try to read logs from VM
        assert self._vm_manager is not None, "Runtime not initialized"
        try:
            result = await self._vm_manager.execute(
                vm,
                f"tail -n {tail} /var/log/agent.log 2>/dev/null || echo 'No logs'",
                timeout=5.0,
            )
            return result.stdout.splitlines() if result.stdout else []
        except Exception:
            return []

    async def execute(
        self,
        agent_id: str,
        code: str,
        language: str = "python",
        timeout: float | None = None,
    ) -> ExecutionResult:
        """
        Execute code in an isolated VM.

        This is a convenience method for one-shot code execution.

        Args:
            agent_id: Agent identifier for tracking
            code: Code to execute
            language: Programming language
            timeout: Execution timeout

        Returns:
            ExecutionResult with output
        """
        if not self._initialized:
            await self.initialize()

        timeout = timeout or self._fc_config.timeout_seconds
        assert self._vm_manager is not None, "Runtime not initialized"

        # Get a VM from pool
        vm = await self._vm_manager.get_or_create_from_pool(self._fc_config)

        try:
            # Write code to VM
            if language == "python":
                code_path = "/tmp/code.py"
                await self._vm_manager.copy_to_vm(vm, _write_temp(code), code_path)
                result = await self._vm_manager.execute(
                    vm,
                    f"python3 {code_path}",
                    timeout=timeout,
                )
            elif language == "bash":
                code_path = "/tmp/code.sh"
                await self._vm_manager.copy_to_vm(vm, _write_temp(code), code_path)
                result = await self._vm_manager.execute(
                    vm,
                    f"bash {code_path}",
                    timeout=timeout,
                )
            else:
                result = ExecutionResult(
                    exit_code=-1,
                    error=f"Unsupported language: {language}",
                )

            return result

        finally:
            # Return VM to pool
            await self._vm_manager.return_to_pool(vm)

    async def _prepare_agent(
        self,
        vm: MicroVM,
        agent_spec: dict[str, Any],
        prompt: str | None,
    ) -> None:
        """Prepare agent code and environment in VM."""
        # Create agent script
        agent_code = self._generate_agent_script(agent_spec, prompt)

        # Write to VM
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(agent_code)
            temp_path = f.name

        assert self._vm_manager is not None, "Runtime not initialized"
        try:
            await self._vm_manager.copy_to_vm(vm, temp_path, "/opt/agent/run.py")
        finally:
            os.unlink(temp_path)

    def _generate_agent_script(
        self,
        agent_spec: dict[str, Any],
        prompt: str | None,
    ) -> str:
        """Generate Python script for agent execution."""
        return f'''#!/usr/bin/env python3
"""
Auto-generated agent script.
"""

import json
import sys

AGENT_SPEC = {json.dumps(agent_spec)}
INITIAL_PROMPT = {json.dumps(prompt)}

def main():
    # Agent execution logic would go here
    print("Agent running with spec:", AGENT_SPEC)
    if INITIAL_PROMPT:
        print("Processing prompt:", INITIAL_PROMPT[:100])

    # Placeholder for actual agent logic
    print("Agent execution complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''

    async def health_check(self) -> dict[str, Any]:
        """Check runtime health."""
        base_health = await super().health_check()

        # Add Firecracker-specific health info
        from runtime.firecracker import FIRECRACKER_AVAILABLE, KVM_AVAILABLE

        base_health.update(
            {
                "firecracker_available": FIRECRACKER_AVAILABLE,
                "kvm_available": KVM_AVAILABLE,
                "binary_path": self._fc_config.firecracker_path,
                "binary_exists": os.path.exists(self._fc_config.firecracker_path),
                "kernel_exists": os.path.exists(self._fc_config.kernel_path),
                "rootfs_exists": os.path.exists(self._fc_config.rootfs_path),
                "active_vms": len(self._active_vms),
                "pool_size": self._pool_size,
            }
        )

        return base_health

    def supports_gpu(self) -> bool:
        """Firecracker does not support GPU passthrough."""
        return False

    def get_resource_limits(self) -> dict[str, Any]:
        """Get current resource limits."""
        return {
            "max_vcpus": self._fc_config.vcpu_count,
            "max_memory_mib": self._fc_config.mem_size_mib,
            "timeout_seconds": self._fc_config.timeout_seconds,
            "max_output_bytes": self._fc_config.max_output_bytes,
        }


def _write_temp(content: str) -> str:
    """Write content to a temporary file and return path."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(content)
        return f.name
