"""CLI tool for wsqlite."""

import json
import sys

import click
from pydantic import BaseModel

from wsqlite import TableSync, WSQLite


def load_model(model_path: str) -> type[BaseModel]:
    """Load a Pydantic model from a Python file."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for item in dir(module):
        obj = getattr(module, item)
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj != BaseModel:
            return obj
    raise ValueError(f"No Pydantic model found in {model_path}")


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """wsqlite - SQLite ORM CLI tool."""
    pass


@cli.command()
@click.argument("db_path")
@click.argument("model_path", type=click.Path(exists=True))
def init(db_path: str, model_path: str):
    """Initialize database table from Pydantic model.

    DB_PATH: Path to SQLite database file.
    MODEL_PATH: Path to Python file with Pydantic model.
    """
    model = load_model(model_path)

    click.echo(f"Creating table '{model.__name__.lower()}'...")
    sync = TableSync(model, db_path)
    sync.create_if_not_exists()
    sync.sync_with_model()
    click.echo("Table created successfully!")


@cli.command()
@click.argument("db_path")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--limit", default=10, help="Limit number of records")
def list(db_path: str, model_path: str, limit: int):
    """List records from a table.

    DB_PATH: Path to SQLite database file.
    MODEL_PATH: Path to Python file with Pydantic model.
    """
    model = load_model(model_path)
    db = WSQLite(model, db_path)

    records = db.get_paginated(limit=limit)

    for record in records:
        click.echo(record.model_dump())


@cli.command()
@click.argument("db_path")
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("data", type=str)
def insert(db_path: str, model_path: str, data: str):
    """Insert a record into the table.

    DB_PATH: Path to SQLite database file.
    MODEL_PATH: Path to Python file with Pydantic model.
    DATA: JSON string with record data.
    """
    model = load_model(model_path)
    db = WSQLite(model, db_path)

    record_data = json.loads(data)
    record = model(**record_data)
    db.insert(record)

    click.echo("Record inserted successfully!")


@cli.command()
@click.argument("db_path")
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("id", type=int)
def get(db_path: str, model_path: str, id: int):
    """Get a record by ID.

    DB_PATH: Path to SQLite database file.
    MODEL_PATH: Path to Python file with Pydantic model.
    ID: Record ID.
    """
    model = load_model(model_path)
    db = WSQLite(model, db_path)

    records = db.get_by_field(id=id)

    if records:
        for record in records:
            click.echo(record.model_dump_json(indent=2))
    else:
        click.echo("Record not found", err=True)
        sys.exit(1)


@cli.command()
@click.argument("db_path")
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("id", type=int)
def delete(db_path: str, model_path: str, id: int):
    """Delete a record by ID.

    DB_PATH: Path to SQLite database file.
    MODEL_PATH: Path to Python file with Pydantic model.
    ID: Record ID.
    """
    model = load_model(model_path)
    db = WSQLite(model, db_path)

    db.delete(id)
    click.echo("Record deleted successfully!")


@cli.command()
@click.argument("db_path")
@click.argument("model_path", type=click.Path(exists=True))
def count(db_path: str, model_path: str):
    """Count records in a table.

    DB_PATH: Path to SQLite database file.
    MODEL_PATH: Path to Python file with Pydantic model.
    """
    model = load_model(model_path)
    db = WSQLite(model, db_path)

    total = db.count()
    click.echo(f"Total records: {total}")


@cli.command()
@click.argument("db_path")
@click.argument("model_path", type=click.Path(exists=True))
def drop(db_path: str, model_path: str):
    """Drop the table.

    DB_PATH: Path to SQLite database file.
    MODEL_PATH: Path to Python file with Pydantic model.
    """
    model = load_model(model_path)

    if click.confirm(
        f"Are you sure you want to drop table '{model.__name__.lower()}'?"
    ):
        sync = TableSync(model, db_path)
        sync.drop_table()
        click.echo("Table dropped successfully!")


@cli.command()
@click.argument("db_path")
def test_connection(db_path: str):
    """Test database connection.

    DB_PATH: Path to SQLite database file.
    """
    try:
        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.close()
        click.echo("Connection successful!")
    except Exception as e:
        click.echo(f"Connection failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
