"""
OpenFaaS Runtime - Serverless agent execution.

Executes agents as serverless functions on OpenFaaS. Best for:
- Burst scaling
- Cost optimization (pay per invocation)
- Event-driven workflows
- Cloud deployments
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from runtime.base import (
    AgentProcess,
    ProcessState,
    Runtime,
    RuntimeConfig,
    RuntimeType,
)

logger = logging.getLogger("titan.runtime.openfaas")


class OpenFaaSRuntime(Runtime):
    """
    OpenFaaS runtime executing agents as serverless functions.

    Features:
    - Auto-scaling based on demand
    - Pay-per-invocation cost model
    - HTTP-triggered functions
    - Async function invocation
    - Kubernetes integration
    """

    type = RuntimeType.OPENFAAS

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        super().__init__(config or RuntimeConfig(type=RuntimeType.OPENFAAS))
        self._gateway_url = config.gateway_url if config else "http://localhost:8080"
        self._openfaas_available = False
        self._http_client: Any = None

    async def initialize(self) -> None:
        """Initialize OpenFaaS runtime."""
        logger.info("Initializing OpenFaaS runtime...")

        # Check OpenFaaS gateway availability
        self._openfaas_available = await self._check_gateway()
        if not self._openfaas_available:
            logger.warning("OpenFaaS gateway not available - runtime disabled")
            self._initialized = False
            return

        self._initialized = True
        logger.info(f"OpenFaaS runtime initialized (gateway: {self._gateway_url})")

    async def _check_gateway(self) -> bool:
        """Check if OpenFaaS gateway is available."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self._gateway_url}/healthz",
                    timeout=5.0,
                )
                return response.status_code == 200
        except Exception as e:
            logger.debug(f"OpenFaaS gateway check failed: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown OpenFaaS runtime."""
        logger.info("Shutting down OpenFaaS runtime...")
        self._initialized = False
        logger.info("OpenFaaS runtime shutdown complete")

    async def spawn(
        self,
        agent_id: str,
        agent_spec: dict[str, Any],
        prompt: str | None = None,
    ) -> AgentProcess:
        """
        Spawn an agent as an OpenFaaS function invocation.

        Args:
            agent_id: Unique identifier for the agent
            agent_spec: Agent specification dict
            prompt: Optional initial prompt

        Returns:
            AgentProcess with invocation info
        """
        if not self._openfaas_available:
            raise RuntimeError("OpenFaaS gateway is not available")

        # Create process record
        process = AgentProcess(
            agent_id=agent_id,
            runtime_type=RuntimeType.OPENFAAS,
            state=ProcessState.STARTING,
            metadata={"spec": agent_spec, "prompt": prompt},
        )
        self._register_process(process)

        # Get function name
        function_name = self._get_function_name(agent_spec)

        logger.info(f"Invoking function: {function_name}")

        try:
            import httpx

            # Prepare payload
            payload = {
                "agent_spec": agent_spec,
                "prompt": prompt,
                "agent_id": agent_id,
            }

            # Invoke function asynchronously
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._gateway_url}/async-function/{function_name}",
                    json=payload,
                    timeout=30.0,
                )

                if response.status_code in (200, 202):
                    # Get invocation ID from header
                    invocation_id = response.headers.get("X-Call-Id", process.process_id)
                    process.metadata["invocation_id"] = invocation_id
                    process.mark_started()
                    logger.info(f"Function invoked: {invocation_id}")

                    # Start monitoring task
                    asyncio.create_task(self._monitor_invocation(process, function_name))
                else:
                    error = response.text
                    logger.error(f"Function invocation failed: {error}")
                    process.mark_failed(f"Invocation failed: {error}")

        except Exception as e:
            logger.error(f"Failed to invoke function: {e}")
            process.mark_failed(str(e))

        return process

    def _get_function_name(self, agent_spec: dict[str, Any]) -> str:
        """Get OpenFaaS function name from agent spec."""
        # Check spec for custom function name
        runtimes = agent_spec.get("spec", {}).get("runtimes", {})
        serverless_config = runtimes.get("serverless", {})

        if "handler" in serverless_config:
            return str(serverless_config["handler"])

        # Check runtime config
        if self.config.function_name:
            return self.config.function_name

        # Build function name from agent name
        agent_name = agent_spec.get("metadata", {}).get("name", "agent")
        return f"titan-{agent_name}"

    async def _monitor_invocation(
        self,
        process: AgentProcess,
        function_name: str,
    ) -> None:
        """Monitor an async function invocation."""
        invocation_id = process.metadata.get("invocation_id")
        if not invocation_id:
            return

        max_wait = self.config.execution_timeout
        elapsed: float = 0
        poll_interval = 2.0

        while process.state == ProcessState.RUNNING and elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            try:
                import httpx

                # Check if result is ready (implementation depends on OpenFaaS setup)
                # This is a simplified polling approach
                async with httpx.AsyncClient() as client:
                    # Try to get result from a hypothetical results endpoint
                    response = await client.get(
                        f"{self._gateway_url}/function/{function_name}/results/{invocation_id}",
                        timeout=5.0,
                    )

                    if response.status_code == 200:
                        result = response.json()
                        process.mark_completed(result)
                        break
                    elif response.status_code == 404:
                        # Still processing
                        continue
                    else:
                        # Error
                        process.mark_failed(f"Result fetch failed: {response.text}")
                        break

            except Exception as e:
                logger.debug(f"Polling error (will retry): {e}")
                continue

        if process.state == ProcessState.RUNNING:
            process.mark_failed("Execution timeout", exit_code=124)

    async def stop(self, process_id: str, force: bool = False) -> bool:
        """
        Stop/cancel an OpenFaaS function invocation.

        Note: OpenFaaS async invocations may not be directly cancellable.
        This marks the process as cancelled locally.
        """
        process = self._processes.get(process_id)
        if not process:
            return False

        process.state = ProcessState.CANCELLED
        process.completed_at = datetime.now()
        logger.info(f"Cancelled invocation tracking for {process_id}")
        return True

    async def get_status(self, process_id: str) -> AgentProcess | None:
        """Get current status of an invocation."""
        return self._processes.get(process_id)

    async def get_logs(self, process_id: str, tail: int = 100) -> list[str]:
        """Get logs from an invocation."""
        process = self._processes.get(process_id)
        if not process:
            return []

        function_name = self._get_function_name(process.metadata.get("spec", {}))

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                # Fetch logs from gateway (if supported)
                response = await client.get(
                    f"{self._gateway_url}/system/functions/{function_name}/logs",
                    params={"tail": tail},
                    timeout=10.0,
                )

                if response.status_code == 200:
                    return response.text.splitlines()[-tail:]

        except Exception as e:
            logger.debug(f"Failed to get logs: {e}")

        return [f"Logs not available for invocation {process_id}"]

    async def health_check(self) -> dict[str, Any]:
        """Check OpenFaaS runtime health."""
        base = await super().health_check()
        base.update(
            {
                "openfaas_available": self._openfaas_available,
                "gateway_url": self._gateway_url,
            }
        )
        return base

    async def deploy_function(
        self,
        agent_name: str,
        image: str,
        env_vars: dict[str, str] | None = None,
    ) -> str:
        """
        Deploy an agent as an OpenFaaS function.

        Args:
            agent_name: Name for the function
            image: Docker image to deploy
            env_vars: Environment variables

        Returns:
            Deployed function name
        """
        if not self._openfaas_available:
            raise RuntimeError("OpenFaaS gateway not available")

        function_name = f"titan-{agent_name}"

        try:
            import httpx

            payload = {
                "service": function_name,
                "image": image,
                "envVars": env_vars or {},
                "labels": {
                    "titan.agent": agent_name,
                    "titan.version": "1.0",
                },
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._gateway_url}/system/functions",
                    json=payload,
                    timeout=60.0,
                )

                if response.status_code in (200, 201, 202):
                    logger.info(f"Deployed function: {function_name}")
                    return function_name
                else:
                    raise RuntimeError(f"Deploy failed: {response.text}")

        except Exception as e:
            logger.error(f"Failed to deploy function: {e}")
            raise

    async def list_functions(self) -> list[dict[str, Any]]:
        """List deployed Titan functions."""
        if not self._openfaas_available:
            return []

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self._gateway_url}/system/functions",
                    timeout=10.0,
                )

                if response.status_code == 200:
                    functions = response.json()
                    # Filter to Titan functions
                    return [f for f in functions if f.get("labels", {}).get("titan.agent")]

        except Exception as e:
            logger.debug(f"Failed to list functions: {e}")

        return []
