"""Connection management for SQLite with thread-safe operations."""

import logging
import threading
from contextlib import contextmanager
from typing import Any, AsyncIterator, Iterator, Optional

logger = logging.getLogger(__name__)

_global_connection_lock = threading.Lock()
_global_connection: Optional[Any] = None
_db_path: Optional[str] = None


def _get_connection(db_path: str) -> Any:
    """Get or create global connection."""
    global _global_connection, _db_path

    with _global_connection_lock:
        if _global_connection is None or _db_path != db_path:
            import sqlite3

            _global_connection = sqlite3.connect(db_path, check_same_thread=False)
            _global_connection.row_factory = sqlite3.Row
            _db_path = db_path
            logger.info(f"Created global connection to {db_path}")
        return _global_connection


def close_global_connection():
    """Close global connection."""
    global _global_connection, _db_path

    with _global_connection_lock:
        if _global_connection:
            _global_connection.close()
            _global_connection = None
            _db_path = None
        logger.info("Global SQLite connection closed")


class _SQLiteConnection:
    """Wrapper for SQLite connection."""

    def __init__(self, conn: Any):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *args):
        pass  # SQLite connections are managed globally

    def __getattr__(self, name):
        return getattr(self._conn, name)


class Transaction:
    """Context manager for database transactions (sync)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[Any] = None
        self._committed = False

    def __enter__(self):
        import sqlite3

        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.isolation_level = None
        self.conn.execute("BEGIN")
        logger.debug("Transaction started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.conn.rollback()
            logger.debug("Transaction rolled back")
        elif not self._committed:
            self.conn.commit()
            logger.debug("Transaction committed")
        self.conn.close()
        return False

    def commit(self):
        self.conn.commit()
        self._committed = True

    def rollback(self):
        self.conn.rollback()

    def execute(self, query: str, values: tuple = None) -> Any:
        cursor = self.conn.cursor()
        cursor.execute(query, values or ())
        if cursor.description:
            return cursor.fetchall()
        return cursor.rowcount


class AsyncTransaction:
    """Context manager for database transactions (async)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[Any] = None
        self._committed = False

    async def __aenter__(self):
        import aiosqlite

        self.conn = await aiosqlite.connect(self.db_path)
        await self.conn.execute("PRAGMA journal_mode=WAL")
        await self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.isolation_level = None
        await self.conn.execute("BEGIN")
        logger.debug("Async transaction started")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self.conn.rollback()
            logger.debug("Async transaction rolled back")
        elif not self._committed:
            await self.conn.commit()
            logger.debug("Async transaction committed")
        await self.conn.close()
        return False

    async def commit(self):
        await self.conn.commit()
        self._committed = True

    async def rollback(self):
        await self.conn.rollback()

    async def execute(self, query: str, values: tuple = None) -> Any:
        cursor = await self.conn.cursor()
        await cursor.execute(query, values or ())
        if cursor.description:
            return await cursor.fetchall()
        return cursor.rowcount


@contextmanager
def get_transaction(db_path: str) -> Iterator[Transaction]:
    """Get a transaction context manager (sync)."""
    with Transaction(db_path) as transaction:
        yield transaction


def get_async_transaction(db_path: str) -> AsyncTransaction:
    """Get an async transaction context manager."""
    return AsyncTransaction(db_path)


def get_connection(db_path: str) -> _SQLiteConnection:
    """Get a connection from global pool (sync).

    Usage:
        with get_connection("database.db") as conn:
            conn.execute(...)
    """
    conn = _get_connection(db_path)
    return _SQLiteConnection(conn)


async def get_async_connection(db_path: str) -> Any:
    """Get a configured async connection.

    Usage:
        async with get_async_connection("database.db") as conn:
            await conn.execute(...)
    """
    import aiosqlite

    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    await conn.executescript("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")
    return conn


def retry_on_lock(max_retries: int = 3, delay: float = 0.1):
    """Decorator to retry operations on database lock.

    Usage:
        @retry_on_lock(max_retries=5, delay=0.2)
        def insert_data(data):
            db.insert(data)
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            import time

            last_error = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    error_msg = str(e).lower()

                    if "locked" in error_msg or "busy" in error_msg:
                        wait_time = delay * (2**attempt)
                        logger.debug(
                            f"Database locked, retry {attempt + 1}/{max_retries} "
                            f"after {wait_time:.2f}s"
                        )
                        time.sleep(wait_time)
                    else:
                        raise

            raise last_error

        return wrapper

    return decorator
