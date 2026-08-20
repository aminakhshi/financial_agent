import os

from alembic import context

config = context.config


def _get_connectable():
    """Reuse the application's engine when invoked programmatically, otherwise
    build one from the same env-var resolution DatabaseManager uses (including
    the SQLite fallback)."""
    connectable = config.attributes.get("connectable")
    if connectable is not None:
        return connectable

    from data.database import DatabaseManager

    fallback = os.getenv("ENABLE_SQLITE_FALLBACK", "true").strip().lower() in {"1", "true", "yes", "on"}
    manager = DatabaseManager(None, use_sqlite_fallback=fallback)
    return manager.engine


def run_migrations_online():
    from data.database import Base

    connectable = _get_connectable()
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=Base.metadata,
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    raise RuntimeError("Offline migrations are not supported; run against a live database.")
run_migrations_online()
