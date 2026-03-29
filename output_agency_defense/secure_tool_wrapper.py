"""
output_agency_defense/secure_tool_wrapper.py
===============================================
Central security layer for all tool calls.

Purpose:
    - ALL tool calls must pass through this wrapper
    - Tools cannot be called directly — only via wrapper
    - Wrapper enforces: authz check → execute → audit log
    - If wrapper is disabled, system MUST NOT work

OWASP Reference:
    - LLM06:2025 Excessive Agency: limit tool permissions, enforce least privilege
    - LLM05:2025 Improper Output Handling: validate tool outputs

Architecture:
    LLM generates tool call
            │
            ▼
    secure_tool_wrapper.invoke()
            │
            ├─► Validate tool is registered
            ├─► Authorize via object_authz_guard (if resource_id present)
            ├─► Execute tool function
            ├─► Audit log (allow/block + evidence)
            │
            ▼
    Result returned to LLM
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    from output_agency_defense.object_authz_guard import ObjectAuthzGuard, Session, AuthzResult
    from output_agency_defense.error_policy import uniform_error, create_error_response, normalize_timing
    from output_agency_defense.resource_registry import ResourceRegistry
except ImportError:
    from object_authz_guard import ObjectAuthzGuard, Session, AuthzResult
    from error_policy import uniform_error, create_error_response, normalize_timing
    from resource_registry import ResourceRegistry


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _FILE_DIR.parent if _FILE_DIR.name == "output_agency_defense" else _FILE_DIR
_LOG_DIR = _PROJECT_ROOT / "logs"


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------
@dataclass
class ToolDefinition:
    """Registered tool with its handler and metadata."""
    name: str
    handler: Callable[..., Any]
    requires_resource: bool = False
    resource_type: Optional[str] = None
    description: str = ""
    allowed_roles: List[str] = field(default_factory=lambda: ["basic", "admin"])


# ---------------------------------------------------------------------------
# Audit log entry
# ---------------------------------------------------------------------------
@dataclass
class AuditEntry:
    """Single audit log record per hocanın spec."""
    timestamp: str
    user_id: str
    tool: str
    args: Dict[str, Any]
    decision: str          # "allow" or "block"
    evidence: List[str]

    def to_jsonl(self) -> str:
        return json.dumps({
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "tool": self.tool,
            "args": self.args,
            "decision": self.decision,
            "evidence": self.evidence,
        }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Secure Tool Wrapper
# ---------------------------------------------------------------------------
class SecureToolWrapper:
    """
    Central security wrapper that intercepts ALL tool calls.

    Key behaviors:
    1. Tools can only be called through this wrapper
    2. Every call goes through authorization check
    3. Every call is audit-logged (allow or block)
    4. If wrapper is not active, no tools can execute

    The wrapper_enabled flag ensures the system cannot work
    if the security layer is bypassed.
    """

    def __init__(
        self,
        registry: ResourceRegistry,
        authz_guard: ObjectAuthzGuard,
        log_path: Optional[Path] = None,
        enabled: bool = True,
    ):
        self.registry = registry
        self.authz_guard = authz_guard
        self.log_path = log_path or (_LOG_DIR / "agency_audit_week1.jsonl")
        self._enabled = enabled
        self._tools: Dict[str, ToolDefinition] = {}

        # Ensure log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Wrapper state
    # ------------------------------------------------------------------
    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def disable(self):
        """Disable the wrapper. ALL tool calls will be blocked."""
        self._enabled = False

    def enable(self):
        """Enable the wrapper."""
        self._enabled = True

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------
    def register_tool(
        self,
        name: str,
        handler: Callable[..., Any],
        requires_resource: bool = False,
        resource_type: Optional[str] = None,
        description: str = "",
        allowed_roles: Optional[List[str]] = None,
    ) -> None:
        """Register a tool that can be invoked through the wrapper."""
        self._tools[name] = ToolDefinition(
            name=name,
            handler=handler,
            requires_resource=requires_resource,
            resource_type=resource_type,
            description=description,
            allowed_roles=allowed_roles or ["basic", "admin"],
        )

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------
    def _write_audit(self, entry: AuditEntry) -> None:
        """Append audit entry to JSONL log file."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(entry.to_jsonl() + "\n")

    # ------------------------------------------------------------------
    # Core: invoke a tool through the security layer
    # ------------------------------------------------------------------
    def invoke(
        self,
        tool_name: str,
        args: Dict[str, Any],
        session: Session,
    ) -> Dict[str, Any]:
        """
        Invoke a tool through the security wrapper.

        This is the ONLY way to call tools. Direct calls are not allowed.

        Args:
            tool_name: Name of the registered tool
            args:      Tool arguments (may include resource_id)
            session:   Current user session

        Returns:
            Dict with "status", "result" or "error", and metadata.
        """
        t0 = time.time()
        evidence = []
        timestamp = datetime.now(timezone.utc).isoformat()

        # --- Check 1: Wrapper must be enabled ---
        if not self._enabled:
            evidence.append("Security wrapper is DISABLED — all calls blocked")
            self._write_audit(AuditEntry(
                timestamp=timestamp, user_id=session.user,
                tool=tool_name, args=args,
                decision="block", evidence=evidence,
            ))
            return {
                "status": "blocked",
                "error": "Security wrapper is disabled. System cannot process tool calls.",
                "evidence": evidence,
            }

        # --- Check 2: Tool must be registered ---
        tool_def = self._tools.get(tool_name)
        if tool_def is None:
            evidence.append(f"Tool '{tool_name}' is not registered")
            self._write_audit(AuditEntry(
                timestamp=timestamp, user_id=session.user,
                tool=tool_name, args=args,
                decision="block", evidence=evidence,
            ))
            return {
                "status": "blocked",
                "error": uniform_error(),
                "evidence": evidence,
            }

        # --- Check 3: Role-based access ---
        if session.role not in tool_def.allowed_roles:
            evidence.append(f"Role '{session.role}' not in allowed roles {tool_def.allowed_roles}")
            self._write_audit(AuditEntry(
                timestamp=timestamp, user_id=session.user,
                tool=tool_name, args=args,
                decision="block", evidence=evidence,
            ))
            normalize_timing(t0)
            return {
                "status": "blocked",
                "error": uniform_error(),
                "evidence": evidence,
            }

        # --- Check 4: Object-level authorization (if tool requires resource) ---
        if tool_def.requires_resource:
            resource_id = args.get("resource_id")
            resource_type = tool_def.resource_type or args.get("resource_type")

            if not resource_id or not resource_type:
                evidence.append("Missing resource_id or resource_type for resource-bound tool")
                self._write_audit(AuditEntry(
                    timestamp=timestamp, user_id=session.user,
                    tool=tool_name, args=args,
                    decision="block", evidence=evidence,
                ))
                normalize_timing(t0)
                return {
                    "status": "blocked",
                    "error": uniform_error(),
                    "evidence": evidence,
                }

            authz_result = self.authz_guard.authorize(resource_type, resource_id, session)
            evidence.extend(authz_result.evidence)

            if not authz_result.is_allowed:
                self._write_audit(AuditEntry(
                    timestamp=timestamp, user_id=session.user,
                    tool=tool_name, args=args,
                    decision="block", evidence=evidence,
                ))
                normalize_timing(t0)
                return {
                    "status": "blocked",
                    "error": uniform_error(),
                    "evidence": evidence,
                }

        # --- All checks passed: execute tool ---
        try:
            result = tool_def.handler(**args)
            evidence.append(f"Tool '{tool_name}' executed successfully")

            self._write_audit(AuditEntry(
                timestamp=timestamp, user_id=session.user,
                tool=tool_name, args=args,
                decision="allow", evidence=evidence,
            ))

            latency_ms = int((time.time() - t0) * 1000)
            return {
                "status": "allowed",
                "result": result,
                "evidence": evidence,
                "latency_ms": latency_ms,
            }

        except Exception as e:
            evidence.append(f"Tool execution error: {str(e)}")
            self._write_audit(AuditEntry(
                timestamp=timestamp, user_id=session.user,
                tool=tool_name, args=args,
                decision="allow", evidence=evidence,
            ))
            return {
                "status": "error",
                "error": str(e),
                "evidence": evidence,
            }


# ---------------------------------------------------------------------------
# Demo tool handlers
# ---------------------------------------------------------------------------
def get_order_details(resource_id: str, **kwargs) -> Dict:
    """Simulated tool: fetch order details."""
    return {"action": "get_order", "resource_id": resource_id, "data": "Order details here"}


def cancel_order(resource_id: str, reason: str = "", **kwargs) -> Dict:
    """Simulated tool: cancel an order."""
    return {"action": "cancel_order", "resource_id": resource_id, "reason": reason}


def get_ticket_info(resource_id: str, **kwargs) -> Dict:
    """Simulated tool: fetch ticket info."""
    return {"action": "get_ticket", "resource_id": resource_id, "data": "Ticket details here"}


def system_status(**kwargs) -> Dict:
    """Simulated tool: check system status (no resource required)."""
    return {"action": "system_status", "status": "healthy", "uptime": "72h"}


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from resource_registry import create_demo_registry

    registry = create_demo_registry()
    authz = ObjectAuthzGuard(registry)

    log_path = Path("logs/agency_audit_week1.jsonl")
    wrapper = SecureToolWrapper(registry, authz, log_path=log_path, enabled=True)

    # Register tools
    wrapper.register_tool("get_order", get_order_details, requires_resource=True, resource_type="order", description="Fetch order details")
    wrapper.register_tool("cancel_order", cancel_order, requires_resource=True, resource_type="order", description="Cancel an order")
    wrapper.register_tool("get_ticket", get_ticket_info, requires_resource=True, resource_type="ticket", description="Fetch ticket info")
    wrapper.register_tool("system_status", system_status, requires_resource=False, description="Check system health")

    # Sessions
    alice = Session(user="user_alice", role="basic")
    bob = Session(user="user_bob", role="basic")

    print(f"{'='*60}")
    print(f"  SECURE TOOL WRAPPER DEMO")
    print(f"  Registered tools: {wrapper.list_tools()}")
    print(f"{'='*60}")

    test_calls = [
        # (description, session, tool, args)
        ("Alice gets her own order", alice, "get_order", {"resource_id": "ORD-001"}),
        ("Bob tries Alice's order (IDOR)", bob, "get_order", {"resource_id": "ORD-001"}),
        ("Bob gets his own order", bob, "get_order", {"resource_id": "ORD-002"}),
        ("Alice cancels her order", alice, "cancel_order", {"resource_id": "ORD-003", "reason": "changed mind"}),
        ("Alice checks system status (no resource)", alice, "system_status", {}),
        ("Bob calls unregistered tool", bob, "delete_user", {"user_id": "user_alice"}),
        ("Alice accesses non-existent order", alice, "get_order", {"resource_id": "ORD-999"}),
    ]

    for desc, session, tool, args in test_calls:
        result = wrapper.invoke(tool, args, session)
        status = result["status"].upper()
        print(f"\n  [{status}] {desc}")
        print(f"    User: {session.user} | Tool: {tool} | Args: {args}")
        if "error" in result:
            print(f"    Error: {result['error']}")
        if "result" in result:
            print(f"    Result: {result['result']}")

    # --- Test: Wrapper disabled ---
    print(f"\n{'='*60}")
    print(f"  WRAPPER DISABLED TEST")
    print(f"{'='*60}")
    wrapper.disable()
    result = wrapper.invoke("system_status", {}, alice)
    print(f"\n  [{result['status'].upper()}] Wrapper disabled → all calls blocked")
    print(f"    Error: {result.get('error')}")

    # Re-enable
    wrapper.enable()

    # Show audit log
    print(f"\n{'='*60}")
    print(f"  AUDIT LOG ({log_path})")
    print(f"{'='*60}")
    if log_path.exists():
        with open(log_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                print(f"  {entry['timestamp'][:19]} | {entry['decision']:5s} | {entry['user_id']:12s} | {entry['tool']}")
