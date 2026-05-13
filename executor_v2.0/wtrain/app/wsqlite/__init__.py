"""wsqlite - SQLite ORM using Pydantic models.

A simple and type-safe SQLite ORM that uses Pydantic models
to define database schemas. Automatically handles table creation,
synchronization, CRUD operations, and connection pooling.

Usage:
    from wsqlite import WSQLite

    class User(BaseModel):
        id: int
        name: str
        email: str

    db = WSQLite(User, "database.db")
    db.insert(User(id=1, name="John", email="john@example.com"))

Async usage:
    await db.insert_async(User(id=1, name="John", email="john@example.com"))
    users = await db.get_all_async()

Connection Pooling:
    db = WSQLite(User, "database.db", pool_size=20)
"""

from wsqlite.builders import QueryBuilder
from wsqlite.models import AuditMixin, SoftDeleteMixin, TimestampMixin
from wsqlite.core.connection import (
    AsyncTransaction,
    Transaction,
    get_async_connection,
    get_async_transaction,
    get_connection,
    get_transaction,
    retry_on_lock,
)
from wsqlite.core.repository import WSQLite as WSQLiteImpl
from wsqlite.core.sync import AsyncTableSync, TableSync
from wsqlite.core.pool import (
    ConnectionPool,
    AsyncConnectionPool,
    get_pool,
    get_async_pool,
    close_pool,
    close_async_pool,
    close_all_pools,
)
from wsqlite.exceptions import (
    WSQLiteError,
    ConnectionError,
    PoolExhaustedError,
    DatabaseLockedError,
    TableSyncError,
    ValidationError,
    OperationError,
    SQLInjectionError,
    TransactionError,
    MigrationError,
    QueryError,
    TimeoutError,
)

__version__ = "1.2.4"

WSQLite = WSQLiteImpl

__all__ = [
    "WSQLite",
    "QueryBuilder",
    "Transaction",
    "AsyncTransaction",
    "get_connection",
    "get_async_connection",
    "get_transaction",
    "get_async_transaction",
    "retry_on_lock",
    "TableSync",
    "AsyncTableSync",
    "ConnectionPool",
    "AsyncConnectionPool",
    "get_pool",
    "get_async_pool",
    "close_pool",
    "close_async_pool",
    "close_all_pools",
    "WSQLiteError",
    "ConnectionError",
    "PoolExhaustedError",
    "DatabaseLockedError",
    "TableSyncError",
    "ValidationError",
    "OperationError",
    "SQLInjectionError",
    "TransactionError",
    "MigrationError",
    "QueryError",
    "TimeoutError",
]

WSQLite = WSQLiteImpl
