"""
output_agency_defense/resource_registry.py
============================================
Dynamic resource registry for object-level authorization.

Purpose:
    - Register different resource types (order, identity, ticket, etc.)
    - Each type has: resource_type, storage_adapter, owner_lookup
    - System works without being tied to a specific data model
    - At least two different resource types must be registerable

OWASP Reference:
    - IDOR Prevention Cheat Sheet
    - LLM06:2025 Excessive Agency
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Storage Adapter Interface
# ---------------------------------------------------------------------------
class StorageAdapter(ABC):
    """Abstract base for resource storage backends."""

    @abstractmethod
    def find(self, resource_id: str) -> Optional[Dict[str, Any]]:
        ...

    @abstractmethod
    def list_all(self) -> List[Dict[str, Any]]:
        ...


class InMemoryStorage(StorageAdapter):
    """In-memory storage for testing and development."""

    def __init__(self, data: Optional[Dict[str, Dict]] = None):
        self._data: Dict[str, Dict[str, Any]] = data or {}

    def find(self, resource_id: str) -> Optional[Dict[str, Any]]:
        return self._data.get(resource_id)

    def list_all(self) -> List[Dict[str, Any]]:
        return list(self._data.values())

    def insert(self, resource_id: str, record: Dict[str, Any]):
        self._data[resource_id] = record

    def delete(self, resource_id: str) -> bool:
        if resource_id in self._data:
            del self._data[resource_id]
            return True
        return False


# ---------------------------------------------------------------------------
# Registry Entry
# ---------------------------------------------------------------------------
@dataclass
class ResourceTypeEntry:
    resource_type: str
    storage: StorageAdapter
    owner_lookup: Callable[[Dict[str, Any]], str]
    description: str = ""


# ---------------------------------------------------------------------------
# Resource Registry
# ---------------------------------------------------------------------------
class ResourceRegistry:
    """
    Central registry for all resource types.
    Decouples authorization from specific data models.
    """

    def __init__(self):
        self._registry: Dict[str, ResourceTypeEntry] = {}

    def register(
        self,
        resource_type: str,
        storage: StorageAdapter,
        owner_lookup: Callable[[Dict[str, Any]], str],
        description: str = "",
    ) -> None:
        if resource_type in self._registry:
            raise ValueError(f"Resource type '{resource_type}' already registered")
        self._registry[resource_type] = ResourceTypeEntry(
            resource_type=resource_type,
            storage=storage,
            owner_lookup=owner_lookup,
            description=description,
        )

    def unregister(self, resource_type: str) -> bool:
        if resource_type in self._registry:
            del self._registry[resource_type]
            return True
        return False

    def is_registered(self, resource_type: str) -> bool:
        return resource_type in self._registry

    def get_entry(self, resource_type: str) -> Optional[ResourceTypeEntry]:
        return self._registry.get(resource_type)

    def list_types(self) -> List[str]:
        return list(self._registry.keys())

    def find(self, resource_type: str, resource_id: str) -> Optional[Dict[str, Any]]:
        entry = self._registry.get(resource_type)
        if entry is None:
            return None
        return entry.storage.find(resource_id)

    def get_owner(self, resource_type: str, resource_id: str) -> Optional[str]:
        entry = self._registry.get(resource_type)
        if entry is None:
            return None
        resource = entry.storage.find(resource_id)
        if resource is None:
            return None
        try:
            return entry.owner_lookup(resource)
        except (KeyError, TypeError):
            return None


# ---------------------------------------------------------------------------
# Demo registry with sample data
# ---------------------------------------------------------------------------
def create_demo_registry() -> ResourceRegistry:
    registry = ResourceRegistry()

    order_storage = InMemoryStorage({
        "ORD-001": {"id": "ORD-001", "owner_id": "user_alice", "product": "Laptop", "amount": 1200.00, "status": "shipped"},
        "ORD-002": {"id": "ORD-002", "owner_id": "user_bob", "product": "Keyboard", "amount": 85.00, "status": "delivered"},
        "ORD-003": {"id": "ORD-003", "owner_id": "user_alice", "product": "Monitor", "amount": 450.00, "status": "processing"},
        "ORD-004": {"id": "ORD-004", "owner_id": "user_charlie", "product": "Mouse", "amount": 35.00, "status": "cancelled"},
    })
    registry.register("order", order_storage, lambda r: r["owner_id"], "Customer orders")

    ticket_storage = InMemoryStorage({
        "TKT-101": {"id": "TKT-101", "assigned_to": "user_alice", "subject": "Login issue", "priority": "high", "status": "open"},
        "TKT-102": {"id": "TKT-102", "assigned_to": "user_bob", "subject": "Billing question", "priority": "medium", "status": "open"},
        "TKT-103": {"id": "TKT-103", "assigned_to": "user_charlie", "subject": "Feature request", "priority": "low", "status": "closed"},
    })
    registry.register("ticket", ticket_storage, lambda r: r["assigned_to"], "Support tickets")

    return registry


if __name__ == "__main__":
    registry = create_demo_registry()
    print(f"Registered types: {registry.list_types()}")
    for rtype, rid in [("order", "ORD-001"), ("order", "ORD-002"), ("ticket", "TKT-101"), ("ticket", "TKT-999")]:
        owner = registry.get_owner(rtype, rid)
        print(f"  {rtype}/{rid} → owner={owner}")
