"""Main repository class for SQLite operations with connection pooling."""

import logging
import re
from typing import Any, Callable, Optional

from pydantic import BaseModel

from wsqlite.core.connection import (
    AsyncTransaction,
    Transaction,
    get_async_connection,
    get_connection,
    get_transaction,
    retry_on_lock,
)
from wsqlite.core.pool import ConnectionPool, get_pool, close_pool
from wsqlite.core.serialization import serialize_value, deserialize_value
from wsqlite.core.sync import AsyncTableSync, TableSync
from wsqlite.exceptions import DatabaseLockedError, SQLInjectionError, TransactionError

logger = logging.getLogger(__name__)


def validate_identifier(identifier: str) -> None:
    """Validate SQL identifier to prevent SQL injection.

    Args:
        identifier: Table or column name to validate.

    Raises:
        SQLInjectionError: If identifier contains dangerous characters.
    """
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
        raise SQLInjectionError(identifier)


class WSQLite:
    """SQLite repository using Pydantic models.

    Provides a simple interface for CRUD operations on SQLite tables,
    with automatic table creation, schema synchronization, and connection pooling.

    Example:
        from pydantic import BaseModel
        from wsqlite import WSQLite

        class User(BaseModel):
            id: int
            name: str
            email: str

        db = WSQLite(User, "database.db")
        db.insert(User(id=1, name="John", email="john@example.com"))
    """

    def __init__(
        self,
        model: type[BaseModel],
        db_path: str,
        pool_size: int = 10,
        min_pool_size: int = 2,
        use_pool: bool = True,
        table_name: Optional[str] = None,
        soft_delete: bool = False,
        deleted_at_field: str = "deleted_at",
        pool: Optional[ConnectionPool] = None,
        sync_handler: Optional[TableSync] = None,
    ):
        """Initialize the repository with a Pydantic model.

        Args:
            model: Pydantic BaseModel class defining the table schema.
            db_path: Path to SQLite database file.
            pool_size: Maximum number of connections in pool (if pool is not provided).
            min_pool_size: Minimum number of connections in pool (if pool is not provided).
            use_pool: Whether to use connection pooling (recommended).
            table_name: Optional custom table name.
            soft_delete: Whether to use soft deletes (default False).
            deleted_at_field: Name of the field for soft deletes (default "deleted_at").
            pool: Optional pre-configured connection pool.
            sync_handler: Optional pre-configured TableSync instance.
        """
        self.model = model
        self.db_path = db_path
        self.table_name = table_name or model.__name__.lower()
        self.use_pool = use_pool
        self.soft_delete = soft_delete
        self.deleted_at_field = deleted_at_field

        if use_pool:
            if pool:
                self._pool = pool
            else:
                self._pool = get_pool(
                    db_path,
                    min_size=min_pool_size,
                    max_size=pool_size,
                )
        else:
            self._pool = None

        self._sync = sync_handler or TableSync(model, db_path, table_name=self.table_name)
        self._sync.create_if_not_exists()
        self._sync.sync_with_model()

        logger.info(
            f"WSQLite initialized for table '{self.table_name}' (pool={use_pool}, size={pool_size}, soft_delete={soft_delete})"
        )

    def _call_hook(self, instance: Any, hook_name: str, *args, **kwargs) -> None:
        """Call a hook method on the model instance if it exists."""
        hook = getattr(instance, hook_name, None)
        if hook and callable(hook):
            hook(*args, **kwargs)

    def _soft_delete_condition(self, prefix: str = "") -> str:
        """Return the SQL condition for soft delete filtering."""
        if not self.soft_delete:
            return ""
        col = f"{prefix}.{self.deleted_at_field}" if prefix else self.deleted_at_field
        return f"{col} IS NULL"

    def _add_soft_delete_filter(self, conditions: str) -> str:
        """Add soft delete condition to WHERE clause if enabled."""
        sd_cond = self._soft_delete_condition()
        if not sd_cond:
            return conditions
        return f"({conditions}) AND {sd_cond}" if conditions else sd_cond

    def _dump(self, data: BaseModel) -> dict:
        """Serialize a model instance to a dictionary for SQLite insertion."""
        data_dict = data.model_dump(mode='json')
        for key, val in data_dict.items():
            if key in self.model.model_fields:
                annotation = self.model.model_fields[key].annotation
                data_dict[key] = serialize_value(val, annotation)
        return data_dict

    def _load(self, row: tuple) -> BaseModel:
        """Deserialize a SQLite row to a model instance."""
        data = {}
        for key, value in zip(self.model.model_fields.keys(), row):
            annotation = self.model.model_fields[key].annotation
            val = deserialize_value(value, annotation) if value is not None else self._default_value(key)
            data[key] = val
        return self.model(**data)

    def _execute(self, query: str, values: tuple = (), commit: bool = True) -> Any:
        """Execute a query using pool or direct connection."""
        if self.use_pool and self._pool:
            with self._pool.connection() as conn:
                cursor = conn.execute(query, values)
                if cursor.description:
                    result = cursor.fetchall()
                elif query.strip().upper().startswith("INSERT"):
                    result = cursor.lastrowid
                else:
                    result = cursor.rowcount
                if commit:
                    conn.commit()
                return result
        else:
            with get_connection(self.db_path) as conn:
                cursor = conn.execute(query, values)
                if cursor.description:
                    result = cursor.fetchall()
                elif query.strip().upper().startswith("INSERT"):
                    result = cursor.lastrowid
                else:
                    result = cursor.rowcount
                if commit:
                    conn.commit()
                return result

    def insert(self, data: BaseModel) -> Any:
        """Insert a new record into the database."""
        self._call_hook(data, "pre_save")
        
        data_dict = self._dump(data)
        fields = ", ".join(data_dict.keys())
        placeholders = ", ".join(["?"] * len(data_dict))
        values = tuple(data_dict.values())

        query = f"INSERT INTO {self.table_name} ({fields}) VALUES ({placeholders})"
        result = self._execute(query, values)
        
        self._call_hook(data, "post_save")
        return result

    def get_all(self) -> list[BaseModel]:
        """Get all records from the table."""
        condition = self._soft_delete_condition()
        where_clause = f" WHERE {condition}" if condition else ""
        query = f"SELECT * FROM {self.table_name}{where_clause}"
        rows = self._execute(query, commit=False)
        return [self._load(row) for row in rows]

    def get_by_field(self, **filters) -> list[BaseModel]:
        """Get records filtered by specified fields."""
        conditions_list = [f"{key} = ?" for key in filters]
        conditions = " AND ".join(conditions_list)
        conditions = self._add_soft_delete_filter(conditions)
        
        where_clause = f" WHERE {conditions}" if conditions else ""
        values = tuple(filters.values())
        query = f"SELECT * FROM {self.table_name}{where_clause}"

        rows = self._execute(query, values, commit=False)
        return [self._load(row) for row in rows]

    def update(self, record_id: int, data: BaseModel) -> None:
        """Update a record in the database."""
        self._call_hook(data, "pre_save")
        
        data_dict = self._dump(data)
        fields = ", ".join(f"{key} = ?" for key in data_dict.keys())
        values = tuple(data_dict.values()) + (record_id,)
        query = f"UPDATE {self.table_name} SET {fields} WHERE id = ?"

        self._execute(query, values)
        self._call_hook(data, "post_save")

    def delete(self, record_id: int) -> None:
        """Delete a record from the database (hard or soft)."""
        if self.soft_delete:
            from datetime import datetime
            now = datetime.now().isoformat()
            query = f"UPDATE {self.table_name} SET {self.deleted_at_field} = ? WHERE id = ?"
            self._execute(query, (now, record_id))
        else:
            query = f"DELETE FROM {self.table_name} WHERE id = ?"
            self._execute(query, (record_id,))

    def restore(self, record_id: int) -> None:
        """Restore a soft-deleted record."""
        if not self.soft_delete:
            return
        query = f"UPDATE {self.table_name} SET {self.deleted_at_field} = NULL WHERE id = ?"
        self._execute(query, (record_id,))

    def load_related(
        self,
        instance: BaseModel,
        attribute_name: str,
        related_db: "WSQLite",
        foreign_key: str,
        local_key: str = "id",
        is_list: bool = True,
    ):
        """Load related data from another table onto a model instance.

        Args:
            instance: The model instance to enrich.
            attribute_name: The name of the attribute to set on the instance.
            related_db: The WSQLite instance for the related table.
            foreign_key: The foreign key column name on the related table.
            local_key: The local key column name on the current instance's table.
            is_list: True for one-to-many relationships, False for many-to-one.
        """
        local_value = getattr(instance, local_key)
        results = related_db.get_by_field(**{foreign_key: local_value})

        if is_list:
            setattr(instance, attribute_name, results)
        else:
            setattr(instance, attribute_name, results[0] if results else None)

    async def load_related_async(
        self,
        instance: BaseModel,
        attribute_name: str,
        related_db: "WSQLite",
        foreign_key: str,
        local_key: str = "id",
        is_list: bool = True,
    ):
        """Load related data from another table onto a model instance (async)."""
        local_value = getattr(instance, local_key)
        results = await related_db.get_by_field_async(**{foreign_key: local_value})

        if is_list:
            setattr(instance, attribute_name, results)
        else:
            setattr(instance, attribute_name, results[0] if results else None)

    def _default_value(self, field: str) -> Any:
        """Get default value for a field when database value is NULL."""
        field_type = self.model.model_fields[field].annotation
        if field_type is str:
            return ""
        elif field_type is int:
            return 0
        elif field_type is bool:
            return False
        return None

    def get_paginated(
        self,
        limit: int = 10,
        offset: int = 0,
        order_by: Optional[str] = None,
        order_desc: bool = False,
    ) -> list[BaseModel]:
        """Get records with pagination."""
        validate_identifier(self.table_name)
        
        condition = self._soft_delete_condition()
        where_clause = f" WHERE {condition}" if condition else ""
        
        if order_by:
            validate_identifier(order_by)
            order_clause = f" ORDER BY {order_by} {'DESC' if order_desc else 'ASC'}"
        else:
            order_clause = ""

        query = f"SELECT * FROM {self.table_name}{where_clause}{order_clause} LIMIT ? OFFSET ?"
        rows = self._execute(query, (limit, offset), commit=False)
        return [self._load(row) for row in rows]

    def get_page(self, page: int = 1, per_page: int = 10) -> list[BaseModel]:
        """Get records by page number.

        Args:
            page: Page number (1-indexed).
            per_page: Number of records per page.

        Returns:
            List of model instances for the requested page.
        """
        if page < 1:
            page = 1
        if per_page < 1:
            per_page = 10
        offset = (page - 1) * per_page
        return self.get_paginated(limit=per_page, offset=offset)

    def count(self) -> int:
        """Get total number of records in the table."""
        validate_identifier(self.table_name)
        condition = self._soft_delete_condition()
        where_clause = f" WHERE {condition}" if condition else ""
        
        query = f"SELECT COUNT(*) FROM {self.table_name}{where_clause}"
        result = self._execute(query, commit=False)
        return result[0][0] if result else 0

    def insert_many(self, data_list: list[BaseModel]) -> None:
        """Insert multiple records in a single transaction.

        Args:
            data_list: List of model instances to insert.
        """
        if not data_list:
            return

        for data in data_list:
            self._call_hook(data, "pre_save")

        data_dicts = [self._dump(data) for data in data_list]
        fields = ", ".join(data_dicts[0].keys())
        placeholders = ", ".join(["?"] * len(data_dicts[0]))

        query = f"INSERT INTO {self.table_name} ({fields}) VALUES ({placeholders})"

        if self.use_pool and self._pool:
            with self._pool.connection() as conn:
                for data_dict in data_dicts:
                    values = tuple(data_dict.values())
                    conn.execute(query, values)
                conn.commit()
        else:
            with get_transaction(self.db_path) as txn:
                for data_dict in data_dicts:
                    values = tuple(data_dict.values())
                    txn.execute(query, values)
                txn.commit()
        
        for data in data_list:
            self._call_hook(data, "post_save")

    def update_many(self, updates: list[tuple[BaseModel, int]]) -> int:
        """Update multiple records.

        Args:
            updates: List of (model, record_id) tuples.

        Returns:
            Number of records updated.
        """
        if not updates:
            return 0

        for data, _ in updates:
            self._call_hook(data, "pre_save")

        validate_identifier(self.table_name)
        total_updated = 0

        if self.use_pool and self._pool:
            with self._pool.connection() as conn:
                for data, record_id in updates:
                    data_dict = self._dump(data)
                    fields = ", ".join(f"{key} = ?" for key in data_dict)
                    values = tuple(data_dict.values()) + (record_id,)
                    query = f"UPDATE {self.table_name} SET {fields} WHERE id = ?"
                    conn.execute(query, values)
                    total_updated += conn.total_changes
                conn.commit()
        else:
            with get_transaction(self.db_path) as txn:
                for data, record_id in updates:
                    data_dict = self._dump(data)
                    fields = ", ".join(f"{key} = ?" for key in data_dict)
                    values = tuple(data_dict.values()) + (record_id,)
                    query = f"UPDATE {self.table_name} SET {fields} WHERE id = ?"
                    txn.execute(query, values)
                    total_updated += txn.conn.total_changes
                txn.commit()

        for data, _ in updates:
            self._call_hook(data, "post_save")

        return total_updated

    def delete_many(self, record_ids: list[int]) -> int:
        """Delete multiple records by their IDs (hard or soft).

        Args:
            record_ids: List of record IDs to delete.

        Returns:
            Number of records deleted.
        """
        if not record_ids:
            return 0

        validate_identifier(self.table_name)

        if self.soft_delete:
            from datetime import datetime
            now = datetime.now().isoformat()
            query = f"UPDATE {self.table_name} SET {self.deleted_at_field} = ? WHERE id = ?"
            params = [(now, rid) for rid in record_ids]
        else:
            query = f"DELETE FROM {self.table_name} WHERE id = ?"
            params = [(rid,) for rid in record_ids]

        if self.use_pool and self._pool:
            with self._pool.connection() as conn:
                for p in params:
                    conn.execute(query, p)
                conn.commit()
        else:
            with get_transaction(self.db_path) as txn:
                for p in params:
                    txn.execute(query, p)
                txn.commit()

        return len(record_ids)

    def execute_transaction(self, operations: list[tuple[str, tuple]]) -> list[Any]:
        """Execute multiple operations in a transaction.

        Args:
            operations: List of (query, params) tuples.

        Returns:
            List of results from each operation.
        """
        results = []
        try:
            if self.use_pool and self._pool:
                with self._pool.connection() as conn:
                    for query, values in operations:
                        cursor = conn.execute(query, values)
                        if cursor.description:
                            results.append(cursor.fetchall())
                    conn.commit()
            else:
                with get_transaction(self.db_path) as txn:
                    for query, values in operations:
                        result = txn.execute(query, values)
                        if result is not None:
                            results.append(result)
                    txn.commit()
            logger.info(f"Transaction completed with {len(operations)} operations")
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            raise TransactionError(f"Transaction failed: {e}") from e
        return results

    def with_transaction(self, func: Callable[[Transaction], Any]) -> Any:
        """Execute a function within a transaction.

        Args:
            func: Function that receives Transaction and performs operations.

        Returns:
            Result of the function.
        """
        try:
            if self.use_pool and self._pool:
                with self._pool.connection() as conn:
                    txn = Transaction(self.db_path)
                    txn.conn = conn
                    result = func(txn)
                    conn.commit()
            else:
                with get_transaction(self.db_path) as txn:
                    result = func(txn)
                    txn.commit()
            logger.info("Transaction completed successfully")
            return result
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            raise TransactionError(f"Transaction failed: {e}") from e

    @retry_on_lock(max_retries=3, delay=0.1)
    def insert_with_retry(self, data: BaseModel) -> None:
        """Insert with automatic retry on database lock."""
        self.insert(data)

    async def insert_async(self, data: BaseModel) -> Any:
        """Insert a new record into the database (async)."""
        self._call_hook(data, "pre_save")
        
        data_dict = self._dump(data)
        fields = ", ".join(data_dict.keys())
        placeholders = ", ".join(["?"] * len(data_dict))
        values = tuple(data_dict.values())

        query = f"INSERT INTO {self.table_name} ({fields}) VALUES ({placeholders})"
        conn = await get_async_connection(self.db_path)
        try:
            cursor = await conn.execute(query, values)
            result = cursor.lastrowid
            await conn.commit()
        finally:
            await conn.close()
            
        self._call_hook(data, "post_save")
        return result

    async def get_all_async(self) -> list[BaseModel]:
        """Get all records from the table (async)."""
        condition = self._soft_delete_condition()
        where_clause = f" WHERE {condition}" if condition else ""
        query = f"SELECT * FROM {self.table_name}{where_clause}"
        conn = await get_async_connection(self.db_path)
        try:
            cursor = await conn.execute(query)
            rows = await cursor.fetchall()
        finally:
            await conn.close()

        return [self._load(row) for row in rows]

    async def get_by_field_async(self, **filters) -> list[BaseModel]:
        """Get records filtered by specified fields (async)."""
        conditions_list = [f"{key} = ?" for key in filters]
        conditions = " AND ".join(conditions_list)
        conditions = self._add_soft_delete_filter(conditions)
        
        where_clause = f" WHERE {conditions}" if conditions else ""
        values = tuple(filters.values())
        query = f"SELECT * FROM {self.table_name}{where_clause}"

        conn = await get_async_connection(self.db_path)
        try:
            cursor = await conn.execute(query, values)
            rows = await cursor.fetchall()
        finally:
            await conn.close()

        return [self._load(row) for row in rows]

    async def update_async(self, record_id: int, data: BaseModel) -> None:
        """Update a record in the database (async)."""
        self._call_hook(data, "pre_save")
        
        data_dict = self._dump(data)
        fields = ", ".join(f"{key} = ?" for key in data_dict.keys())
        values = tuple(data_dict.values()) + (record_id,)
        query = f"UPDATE {self.table_name} SET {fields} WHERE id = ?"

        conn = await get_async_connection(self.db_path)
        try:
            await conn.execute(query, values)
            await conn.commit()
        finally:
            await conn.close()
            
        self._call_hook(data, "post_save")

    async def delete_async(self, record_id: int) -> None:
        """Delete a record from the database (async, hard or soft)."""
        if self.soft_delete:
            from datetime import datetime
            now = datetime.now().isoformat()
            query = f"UPDATE {self.table_name} SET {self.deleted_at_field} = ? WHERE id = ?"
            values = (now, record_id)
        else:
            query = f"DELETE FROM {self.table_name} WHERE id = ?"
            values = (record_id,)
            
        conn = await get_async_connection(self.db_path)
        try:
            await conn.execute(query, values)
            await conn.commit()
        finally:
            await conn.close()

    async def restore_async(self, record_id: int) -> None:
        """Restore a soft-deleted record (async)."""
        if not self.soft_delete:
            return
        query = f"UPDATE {self.table_name} SET {self.deleted_at_field} = NULL WHERE id = ?"
        conn = await get_async_connection(self.db_path)
        try:
            await conn.execute(query, (record_id,))
            await conn.commit()
        finally:
            await conn.close()

    async def search_async(self, query: str, order_by_rank: bool = True) -> list[BaseModel]:
        """Perform a full-text search on an FTS5 table (async).

        Args:
            query: The search query string.
            order_by_rank: Whether to sort results by relevance (default True).

        Returns:
            A list of matching model instances.
        """
        config = getattr(self.model, "wsqlite_config", None)
        if not getattr(config, "use_fts5", False):
            raise OperationError("Search method is only available for FTS5-enabled models.")

        sql = f"SELECT * FROM {self.table_name} WHERE {self.table_name} MATCH ?"
        if order_by_rank:
            sql += " ORDER BY rank"

        conn = await get_async_connection(self.db_path)
        try:
            cursor = await conn.execute(sql, (query,))
            rows = await cursor.fetchall()
        finally:
            await conn.close()

        return [self._load(row) for row in rows]

    async def get_paginated_async(
        self,
        limit: int = 10,
        offset: int = 0,
        order_by: Optional[str] = None,
        order_desc: bool = False,
    ) -> list[BaseModel]:
        """Get records with pagination (async)."""
        validate_identifier(self.table_name)
        
        condition = self._soft_delete_condition()
        where_clause = f" WHERE {condition}" if condition else ""
        
        if order_by:
            validate_identifier(order_by)
            order_clause = f" ORDER BY {order_by} {'DESC' if order_desc else 'ASC'}"
        else:
            order_clause = ""

        query = f"SELECT * FROM {self.table_name}{where_clause}{order_clause} LIMIT ? OFFSET ?"

        conn = await get_async_connection(self.db_path)
        try:
            cursor = await conn.execute(query, (limit, offset))
            rows = await cursor.fetchall()
        finally:
            await conn.close()

        return [self._load(row) for row in rows]

    async def get_page_async(self, page: int = 1, per_page: int = 10) -> list[BaseModel]:
        """Get records by page number (async)."""
        if page < 1:
            page = 1
        if per_page < 1:
            per_page = 10
        offset = (page - 1) * per_page
        return await self.get_paginated_async(limit=per_page, offset=offset)

    async def count_async(self) -> int:
        """Get total number of records in the table (async)."""
        validate_identifier(self.table_name)
        condition = self._soft_delete_condition()
        where_clause = f" WHERE {condition}" if condition else ""
        
        query = f"SELECT COUNT(*) FROM {self.table_name}{where_clause}"
        conn = await get_async_connection(self.db_path)
        try:
            cursor = await conn.execute(query)
            result = await cursor.fetchone()
        finally:
            await conn.close()
        return result[0] if result else 0

    async def insert_many_async(self, data_list: list[BaseModel]) -> None:
        """Insert multiple records in a single transaction (async)."""
        if not data_list:
            return

        for data in data_list:
            self._call_hook(data, "pre_save")

        data_dicts = [self._dump(data) for data in data_list]
        fields = ", ".join(data_dicts[0].keys())
        placeholders = ", ".join(["?"] * len(data_dicts[0]))

        query = f"INSERT INTO {self.table_name} ({fields}) VALUES ({placeholders})"

        conn = await get_async_connection(self.db_path)
        try:
            for data_dict in data_dicts:
                values = tuple(data_dict.values())
                await conn.execute(query, values)
            await conn.commit()
        finally:
            await conn.close()

        for data in data_list:
            self._call_hook(data, "post_save")

    async def update_many_async(self, updates: list[tuple[BaseModel, int]]) -> int:
        """Update multiple records (async)."""
        if not updates:
            return 0

        for data, _ in updates:
            self._call_hook(data, "pre_save")

        validate_identifier(self.table_name)
        total_updated = 0

        conn = await get_async_connection(self.db_path)
        try:
            for data, record_id in updates:
                data_dict = self._dump(data)
                fields = ", ".join(f"{key} = ?" for key in data_dict)
                values = tuple(data_dict.values()) + (record_id,)
                query = f"UPDATE {self.table_name} SET {fields} WHERE id = ?"
                await conn.execute(query, values)
                total_updated += conn.total_changes
            await conn.commit()
        finally:
            await conn.close()

        for data, _ in updates:
            self._call_hook(data, "post_save")

        return total_updated

    async def delete_many_async(self, record_ids: list[int]) -> int:
        """Delete multiple records by their IDs (async, hard or soft)."""
        if not record_ids:
            return 0

        validate_identifier(self.table_name)

        if self.soft_delete:
            from datetime import datetime
            now = datetime.now().isoformat()
            query = f"UPDATE {self.table_name} SET {self.deleted_at_field} = ? WHERE id = ?"
            params = [(now, rid) for rid in record_ids]
        else:
            query = f"DELETE FROM {self.table_name} WHERE id = ?"
            params = [(rid,) for rid in record_ids]

        conn = await get_async_connection(self.db_path)
        try:
            for p in params:
                await conn.execute(query, p)
            await conn.commit()
        finally:
            await conn.close()

        return len(record_ids)

    async def execute_transaction_async(self, operations: list[tuple[str, tuple]]) -> list[Any]:
        """Execute multiple operations in a transaction (async)."""
        results = []
        conn = await get_async_connection(self.db_path)
        try:
            for query, values in operations:
                cursor = await conn.execute(query, values)
                if cursor.description:
                    result = await cursor.fetchall()
                    results.append(result)
            await conn.commit()
            logger.info(f"Async transaction completed with {len(operations)} operations")
        except Exception as e:
            logger.error(f"Async transaction failed: {e}")
            await conn.close()
            raise TransactionError(f"Async transaction failed: {e}") from e
        finally:
            if not conn._connection:  # not closed yet
                await conn.close()
        return results

    async def with_transaction_async(self, func: Callable[[AsyncTransaction], Any]) -> Any:
        """Execute a function within a transaction (async)."""
        try:
            conn = await get_async_connection(self.db_path)
            async with conn:
                txn = AsyncTransaction(self.db_path)
                txn.conn = conn
                result = await func(txn)
                await txn.commit()
                logger.info("Async transaction completed successfully")
                return result
        except Exception as e:
            logger.error(f"Async transaction failed: {e}")
            raise TransactionError(f"Async transaction failed: {e}") from e
