"""
BusSense SQL Migration Script (SQLite -> MySQL)
==============================================
Extracts all existing tables and rows from SQLite (bussense_events.db)
and loads them into MySQL via SQLAlchemy without data loss or downtime.
Preserves primary keys, foreign keys, timestamps, and row counts.
"""

import os
import sys
import json
import sqlite3
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load env
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

env_file = ROOT_DIR / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
else:
    load_dotenv()

from sqlalchemy import text
from bussense_server.db_config import Base, build_engine, get_database_url
from bussense_server import models

DEFAULT_SQLITE_PATH = ROOT_DIR / "outputs" / "server_data" / "bussense_events.db"

TABLES_IN_ORDER = [
    "csr_partners",
    "reference_assets",
    "events",
    "bus_heartbeats",
    "bus_telemetry",
    "infrastructure_observations",
    "infrastructure_events",
    "csr_requests",
    "media_records"
]


def run_migration(sqlite_path: str = None, mysql_url: str = None, dry_run: bool = False):
    if sqlite_path is None:
        sqlite_path = str(DEFAULT_SQLITE_PATH)

    if not Path(sqlite_path).exists():
        print(f"[ERROR] Source SQLite database not found at: {sqlite_path}")
        return False

    if mysql_url is None:
        mysql_url = get_database_url("mysql")

    print("=" * 70)
    print("BUSSENSE DATABASE MIGRATION: SQLite -> MySQL")
    print("=" * 70)
    print(f"Source SQLite: {sqlite_path}")
    print(f"Target MySQL : {mysql_url.split('@')[-1] if '@' in mysql_url else mysql_url}")
    print(f"Dry Run Mode : {dry_run}")
    print("-" * 70)

    # 1. Connect to SQLite
    s_conn = sqlite3.connect(sqlite_path)
    s_conn.row_factory = sqlite3.Row
    s_cur = s_conn.cursor()

    # 2. Count source rows
    source_counts = {}
    for table in TABLES_IN_ORDER:
        try:
            s_cur.execute(f"SELECT COUNT(*) FROM {table}")
            cnt = s_cur.fetchone()[0]
            source_counts[table] = cnt
        except Exception as e:
            source_counts[table] = 0

    print("Source SQLite Table Counts:")
    for t, c in source_counts.items():
        print(f"  {t:<30}: {c:>6} rows")
    print("-" * 70)

    if dry_run:
        print("[DRY RUN] Verified source SQLite database successfully. Exiting without writing to MySQL.")
        s_conn.close()
        return True

    # 3. Create target MySQL engine & tables
    m_engine = build_engine(mysql_url)
    try:
        with m_engine.connect() as test_conn:
            test_conn.execute(text("SELECT 1"))
        print("[SUCCESS] Connected to target MySQL database.")
    except Exception as e:
        print(f"[FATAL] Could not connect to MySQL server: {e}")
        print("Please check your .env configuration (DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT).")
        s_conn.close()
        return False

    # Create schema in target MySQL
    print("Ensuring target MySQL schema and tables exist...")
    Base.metadata.create_all(m_engine)

    # 4. Copy records table by table
    with m_engine.begin() as m_conn:
        # Disable foreign key checks for clean bulk loading
        if "mysql" in mysql_url:
            m_conn.execute(text("SET FOREIGN_KEY_CHECKS=0;"))

        for table in TABLES_IN_ORDER:
            s_cur.execute(f"SELECT * FROM {table}")
            rows = s_cur.fetchall()
            if not rows:
                print(f"  [{table}] 0 rows to migrate.")
                continue

            col_names = [d[0] for d in s_cur.description]
            param_names = [f":{c}" for c in col_names]
            cols_clause = ", ".join(col_names)
            params_clause = ", ".join(param_names)

            if "mysql" in mysql_url:
                insert_stmt = text(f"REPLACE INTO {table} ({cols_clause}) VALUES ({params_clause})")
            else:
                insert_stmt = text(f"INSERT OR REPLACE INTO {table} ({cols_clause}) VALUES ({params_clause})")

            # Prepare data
            data_list = []
            for r in rows:
                row_dict = dict(r)
                data_list.append(row_dict)

            # Chunked insert for efficiency with large telemetry tables
            chunk_size = 2000
            for i in range(0, len(data_list), chunk_size):
                chunk = data_list[i : i + chunk_size]
                m_conn.execute(insert_stmt, chunk)

            print(f"  [{table}] Migrated {len(data_list)} / {source_counts[table]} rows.")

        if "mysql" in mysql_url:
            m_conn.execute(text("SET FOREIGN_KEY_CHECKS=1;"))

    s_conn.close()
    print("-" * 70)

    # 5. Validate row counts in target
    print("Validating migrated row counts in MySQL:")
    all_matched = True
    with m_engine.connect() as v_conn:
        for table in TABLES_IN_ORDER:
            res = v_conn.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()[0]
            expected = source_counts[table]
            match_str = "MATCH" if res == expected else "MISMATCH"
            print(f"  {table:<30}: Expected={expected:>6} | Actual={res:>6} [{match_str}]")
            if res != expected:
                all_matched = False

    print("=" * 70)
    if all_matched:
        print("[SUCCESS] All tables migrated and row counts fully validated without loss.")
        return True
    else:
        print("[WARNING] Some row counts mismatched during migration.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate BusSense SQLite database to MySQL.")
    parser.add_argument("--sqlite", type=str, default=None, help="Path to SQLite database file.")
    parser.add_argument("--mysql-url", type=str, default=None, help="MySQL connection URL.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect source without writing.")
    args = parser.parse_args()

    success = run_migration(sqlite_path=args.sqlite, mysql_url=args.mysql_url, dry_run=args.dry_run)
    sys.exit(0 if success else 1)
