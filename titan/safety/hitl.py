"""
Titan Safety - Human-in-the-Loop Handler

Manages approval requests and responses for high-risk actions.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

from titan.safety.gates import (
    ApprovalGate,
    ApprovalRequest,
    ApprovalResult,
    ApprovalStatus,
    GateRegistry,
    get_gate_registry,
)
from titan.safety.policies import ActionClassifier, ActionPolicy, RiskLevel, get_action_classifier

if TYPE_CHECKING:
    from titan.persistence.audit import AuditLogger

logger = logging.getLogger("titan.safety.hitl")

# Type for approval notification callbacks
ApprovalCallback = Callable[[ApprovalRequest], Coroutine[Any, Any, None]]


@dataclass
class HITLConfig:
    """Configuration for Human-in-the-Loop handler."""

    # Auto-approve low-risk actions
    auto_approve_low_risk: bool = True

    # Default timeout for approval requests
    default_timeout_seconds: int = 300

    # Whether to block on approval (vs async notification)
    blocking: bool = True

    # Redis connection for distributed approval state
    redis_url: str | None = None

    # WebSocket endpoint for real-time notifications
    websocket_endpoint: str | None = None

    # Notification callbacks
    notification_callbacks: list[ApprovalCallback] = field(default_factory=list)


class HITLHandler:
    """
    Human-in-the-Loop approval handler.

    Manages the complete approval workflow:
    1. Receives action requests from agents
    2. Classifies actions by risk
    3. Routes high-risk actions to approval gates
    4. Notifies humans via configured channels
    5. Waits for approval and returns result
    """

    def __init__(
        self,
        config: HITLConfig | None = None,
        classifier: ActionClassifier | None = None,
        gate_registry: GateRegistry | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.config = config or HITLConfig()
        self._classifier = classifier or get_action_classifier()
        self._registry = gate_registry or get_gate_registry()
        self._audit_logger = audit_logger

        # Pending requests tracking
        self._pending: dict[UUID, ApprovalRequest] = {}
        self._callbacks: list[ApprovalCallback] = list(self.config.notification_callbacks)

        # Redis client for distributed state
        self._redis: Any | None = None

    async def start(self) -> None:
        """Start the HITL handler."""
        if self.config.redis_url:
            try:
                import redis.asyncio as redis

                from_url = cast(Any, redis.from_url)
                self._redis = await from_url(self.config.redis_url)
                logger.info("HITL connected to Redis for distributed state")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")

    async def stop(self) -> None:
        """Stop the HITL handler."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    def add_notification_callback(self, callback: ApprovalCallback) -> None:
        """Add a callback to be notified of new approval requests."""
        self._callbacks.append(callback)

    async def check_action(
        self,
        action: str,
        agent_id: str,
        session_id: str,
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[bool, ApprovalResult | None]:
        """
        Check if an action requires approval and handle the workflow.

        Args:
            action: Description of the action
            agent_id: Agent requesting the action
            session_id: Session identifier
            tool_name: Name of tool (if tool call)
            arguments: Tool arguments
            context: Additional context

        Returns:
            Tuple of (approved, result) where result is None if no approval needed
        """
        # Classify the action
        if tool_name:
            policy = self._classifier.classify_tool_call(tool_name, arguments or {})
        else:
            policy = self._classifier.classify(action, context)

        logger.debug(
            f"Action '{action[:50]}...' classified as {policy.risk_level.value} "
            f"(requires_approval={policy.requires_approval})"
        )

        # Check if approval is required
        if not policy.requires_approval:
            if self.config.auto_approve_low_risk and policy.risk_level == RiskLevel.LOW:
                return True, None
            if policy.risk_level == RiskLevel.MEDIUM and not policy.requires_approval:
                return True, None

        # Request approval
        result = await self.request_approval(
            action=action,
            policy=policy,
            agent_id=agent_id,
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments,
            context=context,
        )

        return result.approved, result

    async def request_approval(
        self,
        action: str,
        policy: ActionPolicy,
        agent_id: str,
        session_id: str,
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ApprovalResult:
        """
        Request approval for an action.

        Args:
            action: Description of the action
            policy: Action policy with risk classification
            agent_id: Agent requesting approval
            session_id: Session identifier
            tool_name: Name of tool
            arguments: Tool arguments
            context: Additional context

        Returns:
            ApprovalResult with the decision
        """
        # Get the appropriate gate
        gate = self._registry.get_for_policy(policy)

        # Create approval request
        request = await gate.request_approval(
            action=action,
            agent_id=agent_id,
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments,
            context=context,
        )

        # Store pending request
        self._pending[request.id] = request

        # Store in Redis for distributed access
        if self._redis:
            await self._store_request_in_redis(request)

        # Log to audit
        await self._audit_approval_requested(request)

        # Notify callbacks
        await self._notify_callbacks(request)

        # Wait for approval if blocking
        if self.config.blocking:
            result = await gate.wait_for_approval(
                request.id,
                timeout=policy.timeout_seconds,
            )

            # Clean up
            self._pending.pop(request.id, None)
            if self._redis:
                await self._remove_request_from_redis(request.id)

            # Log result to audit
            await self._audit_approval_response(request, result)

            return result

        # Non-blocking: return pending result
        return ApprovalResult(
            request_id=request.id,
            status=ApprovalStatus.PENDING,
            approved=False,
            reason="Awaiting approval (non-blocking)",
        )

    async def respond_to_approval(
        self,
        request_id: UUID | str,
        approved: bool,
        responder: str | None = None,
        reason: str | None = None,
    ) -> bool:
        """
        Respond to an approval request.

        Args:
            request_id: ID of the request (string or UUID)
            approved: Whether to approve the action
            responder: Who made the decision
            reason: Reason for the decision

        Returns:
            True if response was recorded
        """
        if isinstance(request_id, str):
            request_id = UUID(request_id)

        # Find the request
        request = self._pending.get(request_id)
        if not request:
            # Try to find in registry
            for gate in self._get_all_gates():
                if request_id in gate.get_pending_requests():
                    return gate.respond(request_id, approved, responder, reason)

            logger.warning(f"Request {request_id} not found")
            return False

        # Get the gate and respond
        policy = self._classifier.classify(request.action)
        gate = self._registry.get_for_policy(policy)

        return gate.respond(request_id, approved, responder, reason)

    async def get_pending_requests(self) -> list[ApprovalRequest]:
        """Get all pending approval requests."""
        requests = list(self._pending.values())

        # Also get from Redis if available
        if self._redis:
            redis_requests = await self._get_requests_from_redis()
            # Merge, avoiding duplicates
            existing_ids = {r.id for r in requests}
            for r in redis_requests:
                if r.id not in existing_ids:
                    requests.append(r)

        return sorted(requests, key=lambda r: r.created_at)

    def _get_all_gates(self) -> list[ApprovalGate]:
        """Get all registered gates."""
        gates = []
        for level in RiskLevel:
            gates.append(self._registry.get_for_risk_level(level))
        return gates

    async def _notify_callbacks(self, request: ApprovalRequest) -> None:
        """Notify all registered callbacks of a new request."""
        for callback in self._callbacks:
            try:
                await callback(request)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    async def _store_request_in_redis(self, request: ApprovalRequest) -> None:
        """Store request in Redis for distributed access."""
        if not self._redis:
            return

        try:
            key = f"hitl:request:{request.id}"
            await self._redis.set(
                key,
                json.dumps(request.to_dict()),
                ex=request.timeout_seconds + 60,  # Extra time for cleanup
            )
        except Exception as e:
            logger.warning(f"Failed to store request in Redis: {e}")

    async def _remove_request_from_redis(self, request_id: UUID) -> None:
        """Remove request from Redis."""
        if not self._redis:
            return

        try:
            key = f"hitl:request:{request_id}"
            await self._redis.delete(key)
        except Exception as e:
            logger.warning(f"Failed to remove request from Redis: {e}")

    async def _get_requests_from_redis(self) -> list[ApprovalRequest]:
        """Get all pending requests from Redis."""
        if not self._redis:
            return []

        try:
            keys = await self._redis.keys("hitl:request:*")
            requests = []
            for key in keys:
                data = await self._redis.get(key)
                if data:
                    request = ApprovalRequest.from_dict(json.loads(data))
                    requests.append(request)
            return requests
        except Exception as e:
            logger.warning(f"Failed to get requests from Redis: {e}")
            return []

    async def _audit_approval_requested(self, request: ApprovalRequest) -> None:
        """Log approval request to audit."""
        if not self._audit_logger:
            return

        try:
            await self._audit_logger.log_approval_requested(
                agent_id=request.agent_id,
                session_id=request.session_id,
                action=request.action,
                risk_level=request.risk_level.value,
                context={
                    "request_id": str(request.id),
                    "tool_name": request.tool_name,
                    "arguments": request.arguments,
                    "timeout_seconds": request.timeout_seconds,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to audit approval request: {e}")

    async def _audit_approval_response(
        self,
        request: ApprovalRequest,
        result: ApprovalResult,
    ) -> None:
        """Log approval response to audit."""
        if not self._audit_logger:
            return

        try:
            await self._audit_logger.log_approval_response(
                agent_id=request.agent_id,
                session_id=request.session_id,
                request_id=str(request.id),
                approved=result.approved,
                responder=result.responder,
                reason=result.reason,
            )
        except Exception as e:
            logger.warning(f"Failed to audit approval response: {e}")


# Singleton instance
_default_handler: HITLHandler | None = None


def get_hitl_handler() -> HITLHandler:
    """Get the default HITL handler."""
    global _default_handler
    if _default_handler is None:
        _default_handler = HITLHandler()
    return _default_handler


async def init_hitl_handler(config: HITLConfig | None = None) -> HITLHandler:
    """Initialize and start the HITL handler."""
    global _default_handler
    _default_handler = HITLHandler(config)
    await _default_handler.start()
    return _default_handler
