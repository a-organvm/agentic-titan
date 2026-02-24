"""
Docker Runtime - Containerized agent execution.

Executes agents in isolated Docker containers. Best for:
- Production deployments
- Isolation and security
- Resource limits enforcement
- Reproducible environments
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from datetime import datetime
from typing import Any

from runtime.base import (
    AgentProcess,
    ProcessState,
    Runtime,
    RuntimeConfig,
    RuntimeType,
)

logger = logging.getLogger("titan.runtime.docker")


class DockerRuntime(Runtime):
    """
    Docker runtime executing agents in containers.

    Features:
    - Process isolation
    - Resource limits (CPU, memory)
    - Network isolation options
    - Volume mounting for data persistence
    - Automatic image building
    """

    type = RuntimeType.DOCKER

    # Default base image for agents
    DEFAULT_IMAGE = "python:3.11-slim"
    AGENT_IMAGE_PREFIX = "titan-agent"

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        super().__init__(config or RuntimeConfig(type=RuntimeType.DOCKER))
        self._docker_available = False
        self._default_network = "titan-network"

    async def initialize(self) -> None:
        """Initialize Docker runtime."""
        logger.info("Initializing Docker runtime...")

        # Check Docker availability
        self._docker_available = await self._check_docker()
        if not self._docker_available:
            logger.warning("Docker not available - runtime will fail on spawn")
            self._initialized = False
            return

        # Ensure network exists
        await self._ensure_network()

        self._initialized = True
        logger.info("Docker runtime initialized")

    async def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
            return proc.returncode == 0
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.debug(f"Docker check failed: {e}")
            return False

    async def _ensure_network(self) -> None:
        """Ensure the Docker network exists."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "network",
                "inspect",
                self._default_network,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()

            if proc.returncode != 0:
                # Create network
                proc = await asyncio.create_subprocess_exec(
                    "docker",
                    "network",
                    "create",
                    self._default_network,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
                logger.info(f"Created Docker network: {self._default_network}")
        except Exception as e:
            logger.warning(f"Failed to ensure network: {e}")

    async def shutdown(self) -> None:
        """Shutdown Docker runtime, stopping all containers."""
        logger.info("Shutting down Docker runtime...")

        # Stop all containers
        for process_id in list(self._processes.keys()):
            process = self._processes[process_id]
            if process.container_id and process.state == ProcessState.RUNNING:
                await self.stop(process_id, force=True)

        self._initialized = False
        logger.info("Docker runtime shutdown complete")

    async def spawn(
        self,
        agent_id: str,
        agent_spec: dict[str, Any],
        prompt: str | None = None,
    ) -> AgentProcess:
        """
        Spawn an agent in a Docker container.

        Args:
            agent_id: Unique identifier for the agent
            agent_spec: Agent specification dict
            prompt: Optional initial prompt

        Returns:
            AgentProcess with container info
        """
        if not self._docker_available:
            raise RuntimeError("Docker is not available")

        # Create process record
        process = AgentProcess(
            agent_id=agent_id,
            runtime_type=RuntimeType.DOCKER,
            state=ProcessState.STARTING,
            metadata={"spec": agent_spec, "prompt": prompt},
        )
        self._register_process(process)

        # Determine image
        image = self._get_image(agent_spec)

        # Build docker run command
        container_name = f"titan-{agent_id}-{process.process_id}"
        cmd = self._build_run_command(
            container_name=container_name,
            image=image,
            agent_spec=agent_spec,
            prompt=prompt,
        )

        logger.info(f"Starting container: {container_name}")
        logger.debug(f"Docker command: {' '.join(cmd)}")

        try:
            # Start container using exec (safe - no shell injection possible)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                container_id = stdout.decode().strip()
                process.container_id = container_id
                process.mark_started()
                logger.info(f"Container started: {container_id[:12]}")

                # Start monitoring task
                asyncio.create_task(self._monitor_container(process))
            else:
                error = stderr.decode().strip()
                logger.error(f"Failed to start container: {error}")
                process.mark_failed(f"Container start failed: {error}")

        except Exception as e:
            logger.error(f"Failed to spawn container: {e}")
            process.mark_failed(str(e))

        return process

    def _get_image(self, agent_spec: dict[str, Any]) -> str:
        """Get the Docker image for an agent."""
        # Check spec for custom image
        runtimes = agent_spec.get("spec", {}).get("runtimes", {})
        container_config = runtimes.get("container", {})

        if "image" in container_config:
            return str(container_config["image"])

        # Check runtime config
        if self.config.image:
            return self.config.image

        # Build image name from agent name
        agent_name = agent_spec.get("metadata", {}).get("name", "generic")
        return f"{self.AGENT_IMAGE_PREFIX}-{agent_name}:latest"

    def _build_run_command(
        self,
        container_name: str,
        image: str,
        agent_spec: dict[str, Any],
        prompt: str | None,
    ) -> list[str]:
        """Build the docker run command as argument list (safe from injection)."""
        cmd = [
            "docker",
            "run",
            "-d",  # Detached
            "--name",
            container_name,
            "--network",
            self._default_network,
        ]

        # Resource limits
        if self.config.cpu_limit:
            cmd.extend(["--cpus", str(self.config.cpu_limit)])
        if self.config.memory_limit:
            cmd.extend(["--memory", self.config.memory_limit])

        # GPU support
        if self.config.gpu_required:
            cmd.extend(["--gpus", "all"])

        # Environment variables (passed as separate args, not shell-interpreted)
        env_vars = {
            "TITAN_AGENT_SPEC": json.dumps(agent_spec),
            "TITAN_PROMPT": prompt or "",
            **self.config.environment,
        }
        for key, value in env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Volumes
        for host_path, container_path in self.config.volumes.items():
            cmd.extend(["-v", f"{host_path}:{container_path}"])

        # Auto-remove when done
        cmd.append("--rm")

        # Image
        cmd.append(image)

        # Entry command (run the agent)
        cmd.extend(
            [
                "python",
                "-m",
                "titan.cli",
                "run",
                "--prompt",
                prompt or "",
            ]
        )

        return cmd

    async def _monitor_container(self, process: AgentProcess) -> None:
        """Monitor a container for completion."""
        if not process.container_id:
            return

        while process.state == ProcessState.RUNNING:
            await asyncio.sleep(2.0)

            try:
                # Check if container is still running
                proc = await asyncio.create_subprocess_exec(
                    "docker",
                    "inspect",
                    "-f",
                    "{{.State.Running}}",
                    process.container_id,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()

                if proc.returncode != 0:
                    # Container no longer exists
                    process.mark_failed("Container disappeared")
                    break

                running = stdout.decode().strip().lower() == "true"
                if not running:
                    # Get exit code
                    proc = await asyncio.create_subprocess_exec(
                        "docker",
                        "inspect",
                        "-f",
                        "{{.State.ExitCode}}",
                        process.container_id,
                        stdout=asyncio.subprocess.PIPE,
                    )
                    stdout, _ = await proc.communicate()
                    exit_code = int(stdout.decode().strip())

                    if exit_code == 0:
                        process.mark_completed()
                    else:
                        process.mark_failed(f"Container exited with code {exit_code}", exit_code)
                    break

            except Exception as e:
                logger.error(f"Error monitoring container: {e}")
                process.mark_failed(str(e))
                break

    async def stop(self, process_id: str, force: bool = False) -> bool:
        """
        Stop a container.

        Args:
            process_id: ID of the process
            force: If True, kill instead of stop

        Returns:
            True if stopped
        """
        process = self._processes.get(process_id)
        if not process or not process.container_id:
            return False

        docker_cmd = "kill" if force else "stop"
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                docker_cmd,
                process.container_id,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()

            process.state = ProcessState.CANCELLED
            process.completed_at = datetime.now()
            return True

        except Exception as e:
            logger.error(f"Failed to stop container: {e}")
            return False

    async def get_status(self, process_id: str) -> AgentProcess | None:
        """Get current status of a container process."""
        return self._processes.get(process_id)

    async def get_logs(self, process_id: str, tail: int = 100) -> list[str]:
        """Get logs from a container."""
        process = self._processes.get(process_id)
        if not process or not process.container_id:
            return []

        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "logs",
                "--tail",
                str(tail),
                process.container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            return stdout.decode().splitlines()

        except Exception as e:
            logger.error(f"Failed to get logs: {e}")
            return [f"Error getting logs: {e}"]

    async def health_check(self) -> dict[str, Any]:
        """Check Docker runtime health."""
        base = await super().health_check()
        base.update(
            {
                "docker_available": self._docker_available,
                "network": self._default_network,
            }
        )
        return base

    async def build_agent_image(
        self,
        agent_name: str,
        dockerfile: str | None = None,
        context: str | None = None,
    ) -> str:
        """
        Build a Docker image for an agent.

        Args:
            agent_name: Name of the agent
            dockerfile: Path to Dockerfile (optional)
            context: Build context directory (optional)

        Returns:
            Image name
        """
        image_name = f"{self.AGENT_IMAGE_PREFIX}-{agent_name}:latest"

        # Use default Dockerfile if not provided
        dockerfile_path = dockerfile or self.config.dockerfile
        context = context or self.config.build_context or "."
        temp_dockerfile = None

        if not dockerfile_path:
            # Create a basic Dockerfile inline
            dockerfile_content = f"""
FROM {self.DEFAULT_IMAGE}

WORKDIR /app

# Install Titan
COPY . .
RUN pip install -e .

# Default command
CMD ["python", "-m", "titan.cli", "run"]
"""
            # Secure temp file creation
            temp_dockerfile = tempfile.NamedTemporaryFile(mode="w", delete=False)
            try:
                temp_dockerfile.write(dockerfile_content)
                temp_dockerfile.close()
                dockerfile_path = temp_dockerfile.name
            except Exception as e:
                if temp_dockerfile:
                    os.unlink(temp_dockerfile.name)
                raise e

        try:
            # Build image (using exec, safe from shell injection)
            cmd = ["docker", "build", "-t", image_name, "-f", dockerfile_path, context]
            logger.info(f"Building image: {image_name}")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()

            if proc.returncode != 0:
                raise RuntimeError(f"Failed to build image: {stdout.decode()}")

            logger.info(f"Built image: {image_name}")
            return image_name
        finally:
            # Cleanup temp file if we created one
            if temp_dockerfile and os.path.exists(temp_dockerfile.name):
                os.unlink(temp_dockerfile.name)
