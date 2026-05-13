"""Database migration management for wsqlite.

Provides version-based schema migrations with support for
upgrades, rollbacks, and schema history tracking.

Usage:
    from wsqlite.migrations import MigrationManager, Migration

    manager = MigrationManager("app.db")

    @manager.migration(1, "Create users table")
    def migrate_1():
        # Create initial schema
        pass

    @manager.migration(2, "Add email index")
    def migrate_2():
        # Add new index
        pass

    # Apply all pending migrations
    manager.migrate_up()
"""

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, Union

from wsqlite.exceptions import MigrationError

logger = logging.getLogger(__name__)


@dataclass
class AppliedMigration:
    """Record of an applied migration."""

    version: int
    description: str
    applied_at: str
    duration_ms: Optional[float] = None


@dataclass
class Migration:
    """A database migration."""

    version: int
    description: str
    up: Callable[..., None]
    down: Optional[Callable[..., None]] = None


class MigrationManager:
    """Manages database schema migrations.

    Provides:
    - Version tracking
    - Upgrades and rollbacks
    - Migration history
    - Automatic table creation

    Example:
        manager = MigrationManager("app.db")

        @manager.migration(1, "Create initial schema")
        def m1():
            conn.execute("CREATE TABLE users (...)")

        @manager.migration(2, "Add email column")
        def m2():
            conn.execute("ALTER TABLE users ADD COLUMN email TEXT")

        manager.migrate_up()
    """

    MIGRATIONS_TABLE = "_schema_migrations"

    def __init__(self, db_path: str, pool=None):
        """Initialize migration manager.

        Args:
            db_path: Path to SQLite database.
            pool: Optional connection pool for execution.
        """
        self.db_path = db_path
        self.pool = pool
        self._migrations: dict[int, Migration] = {}
        self._ensure_migrations_table()

    def _ensure_migrations_table(self):
        """Create migrations tracking table if not exists."""
        conn = self._get_connection()
        try:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.MIGRATIONS_TABLE} (
                    version INTEGER PRIMARY KEY,
                    description TEXT NOT NULL,
                    applied_at TEXT NOT NULL,
                    duration_ms REAL
                )
            """)
            conn.commit()
        finally:
            self._return_connection(conn)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        if self.pool:
            return self.pool.get_connection()
        return sqlite3.connect(self.db_path)

    def _return_connection(self, conn: sqlite3.Connection):
        """Return connection to pool or close it."""
        if self.pool:
            self.pool.return_connection(conn)
        else:
            conn.close()

    def migration(
        self,
        version: int,
        description: str,
        allow_down: bool = True,
    ) -> Callable:
        """Decorator to register a migration.

        Args:
            version: Migration version number.
            description: Human-readable description.
            allow_down: Whether to allow rollback.

        Returns:
            Decorator function.

        Example:
            @manager.migration(1, "Create users table")
            def migrate_1(ctx):
                ctx.execute("CREATE TABLE users (...)")
        """

        def decorator(func: Callable[[Any], None]) -> Callable:
            self.register(version, description, func, allow_down)
            return func

        return decorator

    def register(
        self,
        version: int,
        description: str,
        up: Callable[[Any], None],
        down: Optional[Callable[[Any], None]] = None,
    ):
        """Register a migration.

        Args:
            version: Migration version number.
            description: Human-readable description.
            up: Migration function (receives context).
            down: Optional rollback function.
        """
        if version in self._migrations:
            raise MigrationError(f"Migration version {version} already registered")

        self._migrations[version] = Migration(
            version=version,
            description=description,
            up=up,
            down=down,
        )
        logger.debug(f"Registered migration {version}: {description}")

    def get_current_version(self) -> int:
        """Get current schema version.

        Returns:
            Current version number, or 0 if no migrations applied.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(f"SELECT MAX(version) FROM {self.MIGRATIONS_TABLE}")
            result = cursor.fetchone()
            return result[0] if result and result[0] is not None else 0
        finally:
            self._return_connection(conn)

    def get_applied_migrations(self) -> list[AppliedMigration]:
        """Get list of applied migrations.

        Returns:
            List of AppliedMigration objects, ordered by version.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                f"SELECT version, description, applied_at, duration_ms "
                f"FROM {self.MIGRATIONS_TABLE} ORDER BY version"
            )
            return [
                AppliedMigration(
                    version=row[0],
                    description=row[1],
                    applied_at=row[2],
                    duration_ms=row[3],
                )
                for row in cursor.fetchall()
            ]
        finally:
            self._return_connection(conn)

    def get_pending_migrations(self) -> list[Migration]:
        """Get list of migrations not yet applied.

        Returns:
            List of Migration objects to be applied.
        """
        current = self.get_current_version()
        return [m for v, m in sorted(self._migrations.items()) if v > current]

    def migrate_up(self, target_version: Optional[int] = None) -> list[AppliedMigration]:
        """Apply pending migrations up to target version.

        Args:
            target_version: Target version, or None for latest.

        Returns:
            List of applied migrations.
        """
        current = self.get_current_version()
        target = target_version or max(self._migrations.keys(), default=0)

        if target < current:
            raise MigrationError(
                f"Target version {target} is less than current {current}. "
                "Use migrate_down() for rollbacks."
            )

        pending = self.get_pending_migrations()
        applied: list[AppliedMigration] = []

        for migration in pending:
            if migration.version > target:
                break

            logger.info(f"Applying migration {migration.version}: {migration.description}")

            conn = self._get_connection()
            start_time = datetime.now()

            try:

                class MigrationContext:
                    """Context object passed to migration functions."""

                    def __init__(self, conn):
                        self._conn = conn

                    def execute(self, query: str, params: tuple = ()):
                        """Execute a query."""
                        cursor = self._conn.cursor()
                        cursor.execute(query, params)
                        return cursor

                    def executemany(self, query: str, params_list: list):
                        """Execute query with multiple parameter sets."""
                        cursor = self._conn.cursor()
                        cursor.executemany(query, params_list)
                        return cursor

                    def execute_script(self, sql: str):
                        """Execute multiple SQL statements."""
                        self._conn.executescript(sql)

                    @property
                    def connection(self):
                        """Raw connection object."""
                        return self._conn

                ctx = MigrationContext(conn)
                migration.up(ctx)
                conn.commit()

                duration_ms = (datetime.now() - start_time).total_seconds() * 1000

                conn.execute(
                    f"INSERT INTO {self.MIGRATIONS_TABLE} "
                    f"(version, description, applied_at, duration_ms) VALUES (?, ?, ?, ?)",
                    (
                        migration.version,
                        migration.description,
                        datetime.now().isoformat(),
                        duration_ms,
                    ),
                )
                conn.commit()

                applied.append(
                    AppliedMigration(
                        version=migration.version,
                        description=migration.description,
                        applied_at=datetime.now().isoformat(),
                        duration_ms=duration_ms,
                    )
                )

                logger.info(
                    f"Migration {migration.version} applied successfully in {duration_ms:.2f}ms"
                )

            except Exception as e:
                conn.rollback()
                raise MigrationError(
                    f"Migration {migration.version} failed: {e}",
                    version=migration.version,
                    direction="up",
                ) from e

            finally:
                self._return_connection(conn)

        return applied

    def migrate_down(self, target_version: int) -> list[int]:
        """Rollback migrations down to target version.

        Args:
            target_version: Version to rollback to.

        Returns:
            List of rolled back version numbers.
        """
        current = self.get_current_version()

        if target_version > current:
            raise MigrationError(
                f"Target version {target_version} is greater than current {current}. "
                "Use migrate_up() for upgrades."
            )

        applied = self.get_applied_migrations()
        rolled_back: list[int] = []

        for applied_migration in reversed(applied):
            if applied_migration.version <= target_version:
                break

            migration = self._migrations.get(applied_migration.version)
            if migration is None:
                raise MigrationError(
                    f"No migration found for applied version {applied_migration.version}. "
                    "Database may be in inconsistent state."
                )

            if migration.down is None:
                raise MigrationError(
                    f"Migration {migration.version} has no down function. Cannot rollback."
                )

            logger.info(f"Rolling back migration {migration.version}: {migration.description}")

            conn = self._get_connection()

            try:

                class MigrationContext:
                    def __init__(self, conn):
                        self._conn = conn

                    def execute(self, query: str, params: tuple = ()):
                        cursor = self._conn.cursor()
                        cursor.execute(query, params)
                        return cursor

                    def execute_script(self, sql: str):
                        self._conn.executescript(sql)

                    @property
                    def connection(self):
                        return self._conn

                ctx = MigrationContext(conn)
                migration.down(ctx)
                conn.commit()

                conn.execute(
                    f"DELETE FROM {self.MIGRATIONS_TABLE} WHERE version = ?", (migration.version,)
                )
                conn.commit()

                rolled_back.append(migration.version)
                logger.info(f"Migration {migration.version} rolled back successfully")

            except Exception as e:
                conn.rollback()
                raise MigrationError(
                    f"Rollback of migration {migration.version} failed: {e}",
                    version=migration.version,
                    direction="down",
                ) from e

            finally:
                self._return_connection(conn)

        return rolled_back

    def reset(self):
        """Reset all migrations (dangerous!).

        This drops the migrations table and all data.
        Use with extreme caution.
        """
        conn = self._get_connection()
        try:
            conn.execute(f"DROP TABLE IF EXISTS {self.MIGRATIONS_TABLE}")
            conn.commit()
            logger.warning("All migration history reset")
        finally:
            self._return_connection(conn)

    def status(self) -> dict:
        """Get migration status.

        Returns:
            Dictionary with current state.
        """
        current = self.get_current_version()
        pending = self.get_pending_migrations()
        applied = self.get_applied_migrations()

        return {
            "current_version": current,
            "latest_version": max(self._migrations.keys(), default=0),
            "pending_count": len(pending),
            "applied_count": len(applied),
            "is_up_to_date": len(pending) == 0,
            "migrations": {
                "applied": [a.version for a in applied],
                "pending": [p.version for p in pending],
                "registered": list(self._migrations.keys()),
            },
        }


def create_migration_manager(
    db_path: str,
    migrations: list[tuple[int, str, Callable, Optional[Callable]]],
) -> MigrationManager:
    """Factory function to create a migration manager with pre-registered migrations.

    Args:
        db_path: Database path.
        migrations: List of (version, description, up_func, down_func) tuples.

    Returns:
        Configured MigrationManager.
    """
    manager = MigrationManager(db_path)

    for version, description, up, down in migrations:
        manager.register(version, description, up, down)

    return manager
