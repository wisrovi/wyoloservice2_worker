"""Thread-safe connection pool for SQLite with WAL mode optimization."""

import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from queue import Queue, Empty
from threading import Lock, Semaphore
from time import time
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


class PoolExhaustedError(Exception):
    """Raised when connection pool is exhausted."""

    def __init__(self, max_size: int, timeout: float):
        self.max_size = max_size
        self.timeout = timeout
        super().__init__(
            f"Connection pool exhausted (max={max_size}). "
            f"Timeout after {timeout}s. Consider increasing pool size."
        )


class DatabaseLockedError(Exception):
    """Raised when database is locked by another connection."""

    def __init__(self, operation: str):
        self.operation = operation
        super().__init__(
            f"Database locked during '{operation}'. Consider retrying or increasing busy_timeout."
        )


class ConnectionPool:
    """High-performance thread-safe connection pool for SQLite.

    Features:
    - Thread-safe with minimal lock contention
    - WAL mode enabled by default for concurrent read/write
    - Connection health checks
    - Configurable pool size and timeouts
    - Automatic connection recycling

    Example:
        pool = ConnectionPool("app.db", min_size=2, max_size=20)
        with pool.get_connection() as conn:
            conn.execute("SELECT * FROM users")
    """

    def __init__(
        self,
        db_path: str,
        min_size: int = 2,
        max_size: int = 20,
        timeout: float = 30.0,
        wal_mode: bool = True,
        busy_timeout: int = 5000,
        cache_size: int = -2000,
        synchronous: str = "NORMAL",
        journal_mode: str = "WAL",
        foreign_keys: bool = True,
    ):
        self.db_path = db_path
        self.min_size = min_size
        self.max_size = max_size
        self.timeout = timeout
        self.wal_mode = wal_mode
        self.busy_timeout = busy_timeout
        self.cache_size = cache_size
        self.synchronous = synchronous
        self.journal_mode = journal_mode
        self.foreign_keys = foreign_keys

        self._pool: Queue = Queue(maxsize=max_size)
        self._size = 0
        self._lock = Lock()
        self._semaphore = Semaphore(max_size)
        self._created = False
        self._closed = False
        self._pool_id = str(uuid.uuid4())[:8]

        self._init_pragmas()
        self._initialize_min_connections()

        logger.info(
            f"Pool[{self._pool_id}] initialized: {db_path}, "
            f"min={min_size}, max={max_size}, timeout={timeout}s"
        )

    def _init_pragmas(self):
        """Initialize database with optimal pragmas."""
        conn = self._create_connection()
        try:
            if self.wal_mode:
                conn.execute(f"PRAGMA journal_mode={self.journal_mode}")
            conn.execute(f"PRAGMA synchronous={self.synchronous}")
            conn.execute(f"PRAGMA busy_timeout={self.busy_timeout}")
            conn.execute(f"PRAGMA cache_size={self.cache_size}")
            if self.foreign_keys:
                conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")
            conn.commit()
        finally:
            conn.close()

    def _initialize_min_connections(self):
        """Create minimum number of connections."""
        for _ in range(self.min_size):
            conn = self._create_connection()
            self._pool.put(conn)
            self._size += 1
        self._created = True

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with optimal settings."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            check_same_thread=False,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row

        conn.execute(f"PRAGMA journal_mode={self.journal_mode}")
        conn.execute(f"PRAGMA synchronous={self.synchronous}")
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout}")
        conn.execute(f"PRAGMA cache_size={self.cache_size}")
        if self.foreign_keys:
            conn.execute("PRAGMA foreign_keys = ON")

        return conn

    def get_connection(self, timeout: Optional[float] = None) -> sqlite3.Connection:
        """Get a connection from the pool.

        Args:
            timeout: Maximum time to wait for a connection.

        Returns:
            sqlite3.Connection from the pool.

        Raises:
            PoolExhaustedError: If no connection available within timeout.
        """
        if self._closed:
            raise RuntimeError("Pool is closed")

        timeout = timeout or self.timeout
        acquired_time = time()

        if not self._semaphore.acquire(timeout=timeout):
            raise PoolExhaustedError(self.max_size, timeout)

        try:
            try:
                conn = self._pool.get_nowait()
            except Empty:
                with self._lock:
                    if self._size < self.max_size:
                        conn = self._create_connection()
                        self._size += 1
                        logger.debug(
                            f"Pool[{self._pool_id}] created new connection ({self._size}/{self.max_size})"
                        )
                    else:
                        self._semaphore.release()
                        raise PoolExhaustedError(self.max_size, timeout)

            if not self._is_healthy(conn):
                logger.debug(f"Pool[{self._pool_id}] replacing unhealthy connection")
                try:
                    conn.close()
                except:
                    pass
                conn = self._create_connection()

            wait_time = time() - acquired_time
            if wait_time > 0.1:
                logger.debug(f"Pool[{self._pool_id}] waited {wait_time:.3f}s for connection")

            return conn

        except Exception:
            self._semaphore.release()
            raise

    def return_connection(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool.

        Args:
            conn: Connection to return.
        """
        try:
            conn.rollback()

            if self._is_healthy(conn):
                try:
                    self._pool.put_nowait(conn)
                except:
                    conn.close()
                    with self._lock:
                        self._size -= 1
            else:
                conn.close()
                with self._lock:
                    self._size -= 1
                logger.debug(f"Pool[{self._pool_id}] removed unhealthy connection")

        finally:
            self._semaphore.release()

    def _is_healthy(self, conn: sqlite3.Connection) -> bool:
        """Check if connection is still valid and usable."""
        try:
            cursor = conn.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return True
        except (sqlite3.OperationalError, sqlite3.ProgrammingError):
            return False

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for getting a pooled connection.

        Example:
            with pool.connection() as conn:
                conn.execute("SELECT * FROM users")
        """
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.return_connection(conn)

    def execute(
        self,
        query: str,
        params: tuple = (),
        commit: bool = True,
    ) -> Optional[list]:
        """Execute a query using pooled connection.

        Args:
            query: SQL query.
            params: Query parameters.
            commit: Whether to commit after execution.

        Returns:
            Query results if SELECT, rowcount otherwise.
        """
        with self.connection() as conn:
            cursor = conn.execute(query, params)
            if cursor.description:
                result = cursor.fetchall()
            else:
                result = cursor.rowcount

            if commit:
                conn.commit()

            return result

    def close_all(self) -> None:
        """Close all connections in pool."""
        if self._closed:
            return

        self._closed = True
        closed = 0

        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
                closed += 1
            except Empty:
                break

        with self._lock:
            self._size = 0

        logger.info(f"Pool[{self._pool_id}] closed {closed} connections")

    def __len__(self) -> int:
        """Return number of connections in pool."""
        return self._size

    def __enter__(self) -> "ConnectionPool":
        return self

    def __exit__(self, *args) -> None:
        self.close_all()

    @property
    def stats(self) -> dict:
        """Get pool statistics."""
        return {
            "pool_id": self._pool_id,
            "db_path": self.db_path,
            "current_size": self._size,
            "min_size": self.min_size,
            "max_size": self.max_size,
            "available": self.max_size - self._size,
            "closed": self._closed,
        }


class AsyncConnectionPool:
    """Async connection pool for aiosqlite.

    Example:
        pool = AsyncConnectionPool("app.db")
        async with pool.connection() as conn:
            await conn.execute("SELECT * FROM users")
    """

    def __init__(
        self,
        db_path: str,
        min_size: int = 2,
        max_size: int = 20,
        foreign_keys: bool = True,
    ):
        import aiosqlite

        self.db_path = db_path
        self.min_size = min_size
        self.max_size = max_size
        self.foreign_keys = foreign_keys
        self._pool: Queue = Queue(maxsize=max_size)
        self._size = 0
        self._lock = Lock()
        self._aiosqlite = aiosqlite
        self._closed = False
        self._pool_id = str(uuid.uuid4())[:8]

        self._init_pragmas()
        self._initialize_min_connections()

        logger.info(f"AsyncPool[{self._pool_id}] initialized: {db_path}")

    def _init_pragmas(self):
        """Initialize database with optimal pragmas."""
        import asyncio

        async def _init():
            conn = await self._aiosqlite.connect(self.db_path)
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute("PRAGMA busy_timeout=5000")
            await conn.execute("PRAGMA cache_size=-2000")
            if self.foreign_keys:
                await conn.execute("PRAGMA foreign_keys = ON")
            await conn.commit()
            await conn.close()

        asyncio.run(_init())

    def _initialize_min_connections(self):
        """Create minimum number of connections."""
        import asyncio

        async def _create():
            for _ in range(self.min_size):
                conn = await self._aiosqlite.connect(self.db_path)
                conn.row_factory = self._aiosqlite.Row
                if self.foreign_keys:
                    await conn.execute("PRAGMA foreign_keys = ON")
                self._pool.put(conn)
                self._size += 1

        asyncio.run(_create())

    async def _create_connection(self):
        """Create a new async connection."""
        conn = await self._aiosqlite.connect(self.db_path)
        conn.row_factory = self._aiosqlite.Row
        if self.foreign_keys:
            await conn.execute("PRAGMA foreign_keys = ON")
        return conn

    async def get_connection(self):
        """Get a connection from the pool."""
        if self._closed:
            raise RuntimeError("Pool is closed")

        try:
            conn = self._pool.get_nowait()
        except Empty:
            if self._size < self.max_size:
                conn = await self._create_connection()
                self._size += 1
            else:
                raise PoolExhaustedError(self.max_size, 30.0)

        return conn

    async def return_connection(self, conn) -> None:
        """Return a connection to the pool."""
        try:
            await conn.execute("ROLLBACK")

            if self._size > self.min_size:
                await conn.close()
                self._size -= 1
            else:
                self._pool.put_nowait(conn)
        except:
            await conn.close()
            self._size -= 1

    @contextmanager
    async def connection(self):
        """Async context manager for pooled connection."""
        conn = await self.get_connection()
        try:
            yield conn
        finally:
            await self.return_connection(conn)

    async def close_all(self) -> None:
        """Close all connections."""
        if self._closed:
            return

        self._closed = True

        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                await conn.close()
            except Empty:
                break

        logger.info(f"AsyncPool[{self._pool_id}] closed")


_global_sync_pool: Optional[ConnectionPool] = None
_global_async_pool: Optional[AsyncConnectionPool] = None
_global_pool_lock = Lock()
_global_async_lock = Lock()


def get_pool(
    db_path: str,
    min_size: int = 2,
    max_size: int = 10,
    **kwargs,
) -> ConnectionPool:
    """Get or create global synchronous connection pool.

    Args:
        db_path: Path to SQLite database.
        min_size: Minimum pool size.
        max_size: Maximum pool size.
        **kwargs: Additional pool configuration.

    Returns:
        Global ConnectionPool instance.
    """
    global _global_sync_pool

    with _global_pool_lock:
        if _global_sync_pool is None or _global_sync_pool.db_path != db_path:
            if _global_sync_pool is not None:
                _global_sync_pool.close_all()
            _global_sync_pool = ConnectionPool(
                db_path,
                min_size=min_size,
                max_size=max_size,
                **kwargs,
            )
        return _global_sync_pool


def get_async_pool(
    db_path: str,
    min_size: int = 2,
    max_size: int = 10,
    **kwargs,
) -> AsyncConnectionPool:
    """Get or create global asynchronous connection pool."""
    global _global_async_pool

    with _global_async_lock:
        if _global_async_pool is None or _global_async_pool.db_path != db_path:
            _global_async_pool = AsyncConnectionPool(
                db_path,
                min_size=min_size,
                max_size=max_size,
                **kwargs,
            )
        return _global_async_pool


def close_pool() -> None:
    """Close global synchronous pool."""
    global _global_sync_pool
    with _global_pool_lock:
        if _global_sync_pool:
            _global_sync_pool.close_all()
            _global_sync_pool = None


async def close_async_pool() -> None:
    """Close global asynchronous pool."""
    global _global_async_pool
    with _global_async_lock:
        if _global_async_pool:
            await _global_async_pool.close_all()
            _global_async_pool = None


def close_all_pools() -> None:
    """Close all global pools."""
    close_pool()
