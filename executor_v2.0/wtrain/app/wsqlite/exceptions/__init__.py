"""Custom exceptions for wsqlite library."""


class WSQLiteError(Exception):
    """Base exception for wsqlite library.

    All other wsqlite exceptions inherit from this class.
    """

    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ConnectionError(WSQLiteError):
    """Exception raised for connection-related errors.

    Examples:
        - Database file not found
        - Permission denied
        - Invalid connection parameters
    """

    pass


class PoolExhaustedError(ConnectionError):
    """Raised when connection pool is exhausted and all connections are in use.

    This typically indicates:
    - Pool size too small for the workload
    - Connections not being returned to pool properly
    - Long-running queries blocking connections

    Suggestions:
        - Increase pool max_size
        - Use connection context managers properly
        - Optimize slow queries
        - Reduce transaction scope
    """

    def __init__(self, max_size: int, timeout: float = 30.0):
        self.max_size = max_size
        self.timeout = timeout
        message = (
            f"Connection pool exhausted (max={max_size} connections). "
            f"Timeout after {timeout}s. "
            "Consider: increasing pool size, reducing contention, or optimizing queries."
        )
        super().__init__(message, {"max_size": max_size, "timeout": timeout})


class DatabaseLockedError(ConnectionError):
    """Raised when database is locked by another connection or process.

    SQLite uses database-level locking. This error occurs when:
    - Multiple processes trying to write simultaneously
    - Long-running write transactions
    - Unclosed connections from previous operations

    Suggestions:
        - Retry with exponential backoff
        - Use WAL mode (enabled by default)
        - Reduce transaction size
        - Ensure connections are properly closed
    """

    def __init__(self, operation: str, query: str = None):
        self.operation = operation
        self.query = query
        message = (
            f"Database locked during '{operation}'. "
            "Another process may be writing. Consider retrying."
        )
        super().__init__(message, {"operation": operation, "query": query})


class TableSyncError(WSQLiteError):
    """Exception raised during table synchronization operations.

    Examples:
        - Failed to create table
        - Column type mismatch during sync
        - Invalid index creation
    """

    pass


class ValidationError(WSQLiteError):
    """Exception raised for data validation failures.

    Examples:
        - Invalid field type for Pydantic model
        - Missing required field
        - Field validation rule violation
    """

    pass


class OperationError(WSQLiteError):
    """Exception raised for general database operation failures.

    This is a catch-all for database errors that don't fit other categories.

    Examples:
        - Query syntax error
        - Constraint violation
        - Foreign key violation
        - Unique constraint violation
    """

    def __init__(self, message: str, query: str = None, params: tuple = None):
        self.query = query
        self.params = params
        details = {"query": query, "params": params} if query else {}
        super().__init__(message, details)


class SQLInjectionError(WSQLiteError):
    """Raised when potential SQL injection is detected.

    This is a security feature. The library validates all identifiers
    (table names, column names) to prevent SQL injection attacks.

    Only alphanumeric characters and underscores are allowed in identifiers.
    """

    def __init__(self, identifier: str):
        self.identifier = identifier
        message = (
            f"Invalid or potentially dangerous SQL identifier detected: '{identifier}'. "
            "Identifiers must start with a letter or underscore and contain only "
            "alphanumeric characters and underscores."
        )
        super().__init__(message, {"identifier": identifier})


class TransactionError(WSQLiteError):
    """Exception raised for transaction-related errors.

    Examples:
        - Transaction already committed
        - Transaction already rolled back
        - Nested transactions not supported
        - Savepoint errors
    """

    pass


class MigrationError(WSQLiteError):
    """Exception raised during database migration operations.

    Examples:
        - Migration version conflict
        - Failed to apply migration
        - Rollback failed
        - Missing migration file
    """

    def __init__(self, message: str, version: int = None, direction: str = None):
        self.version = version
        self.direction = direction
        details = {"version": version, "direction": direction} if version else {}
        super().__init__(message, details)


class QueryError(WSQLiteError):
    """Exception raised for query execution errors.

    This includes:
        - Syntax errors
        - Type errors
        - Constraint violations
        - Missing tables/columns
    """

    def __init__(self, message: str, query: str = None, original_error: Exception = None):
        self.query = query
        self.original_error = original_error
        details = {"query": query}
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(message, details)


class TimeoutError(WSQLiteError):
    """Exception raised when an operation times out.

    This can occur when:
        - Query takes too long
        - Lock acquisition times out
        - Connection pool times out
    """

    def __init__(self, operation: str, timeout: float):
        self.operation = operation
        self.timeout = timeout
        message = (
            f"Operation '{operation}' timed out after {timeout}s. "
            "Consider optimizing the query or increasing the timeout."
        )
        super().__init__(message, {"operation": operation, "timeout": timeout})


__all__ = [
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
