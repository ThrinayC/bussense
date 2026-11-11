"""
BusSense Database Engine & Session Factory (SQLAlchemy)
======================================================
Provides flexible backend database connectivity supporting MySQL (default production)
and SQLite (fallback/testing), with connection pooling, pool_pre_ping, and session lifecycle.
"""

import os
from pathlib import Path
from typing import Generator
from urllib.parse import quote_plus
from dotenv import load_dotenv

from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import QueuePool, StaticPool

# Load environment variables from .env if present
ROOT_DIR = Path(__file__).resolve().parent.parent
env_path = ROOT_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

DB_DIALECT = os.getenv("DB_DIALECT", "mysql").lower()
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "bussense")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))

Base = declarative_base()


def get_database_url(dialect: str = None) -> str:
    """Builds the database URL based on dialect and environment parameters."""
    if dialect is None:
        dialect = DB_DIALECT

    if dialect == "mysql":
        # PyMySQL driver with URL-encoded credentials and utf8mb4 charset
        user_enc = quote_plus(DB_USER)
        pwd_enc = f":{quote_plus(DB_PASSWORD)}" if DB_PASSWORD else ""
        return f"mysql+pymysql://{user_enc}{pwd_enc}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
    else:
        # SQLite local fallback
        default_sqlite = ROOT_DIR / "outputs" / "server_data" / "bussense_events.db"
        default_sqlite.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{default_sqlite}"


def build_engine(db_url: str = None, is_testing: bool = False):
    """
    Constructs an SQLAlchemy engine with sensible connection pooling
    and pre-ping health checks.
    """
    if db_url is None:
        db_url = get_database_url()

    if db_url.startswith("sqlite"):
        if ":memory:" in db_url or is_testing:
            engine = create_engine(
                db_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        else:
            engine = create_engine(
                db_url,
                connect_args={"check_same_thread": False},
                pool_pre_ping=True,
            )

        # Enable foreign key enforcement for SQLite
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    else:
        # MySQL engine with QueuePool
        engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=DB_POOL_SIZE,
            max_overflow=DB_MAX_OVERFLOW,
            pool_recycle=DB_POOL_RECYCLE,
            pool_timeout=DB_POOL_TIMEOUT,
            pool_pre_ping=True,
            echo=False,
        )

    return engine


# Global engine and sessionmaker singleton instances
engine = build_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency for yielding database sessions with automatic
    rollback on uncaught exceptions and cleanup.
    """
    session = SessionLocal()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
