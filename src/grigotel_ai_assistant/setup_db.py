"""
GrigoTel AI Assistant - Step 1: DuckDB setup

Creates the GrigoTel DuckDB database with two tables and loads them from the CSVs:
  - Stock            (one row per model + color variant)
  - Inventory_Stock  (current quantity per item_id)

Running it again drops and rebuilds both tables, so it also resets the stock.

Location: src/grigotel_ai_assistant/setup_db.py

Usage (from the project root):
    uv run src/grigotel_ai_assistant/setup_db.py
"""

from pathlib import Path

import duckdb

# This file lives in src/grigotel_ai_assistant/, so the project root is 2 levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_DIR = PROJECT_ROOT / "GrigoTel_DB"

DB_PATH = DB_DIR / "grigotel.duckdb"
STOCK_CSV = DB_DIR / "Stock.csv"
INVENTORY_CSV = DB_DIR / "inventory_stock.csv"


def create_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Drop and recreate both tables with explicit types."""
    # Child table first, because it references Stock
    con.execute("DROP TABLE IF EXISTS Inventory_Stock")
    con.execute("DROP TABLE IF EXISTS Stock")

    con.execute("""
        CREATE TABLE Stock (
            item_id       VARCHAR PRIMARY KEY,
            model         VARCHAR NOT NULL,
            color         VARCHAR NOT NULL,
            price         DECIMAL(10, 2) NOT NULL,
            release_date  DATE
        )
    """)

    con.execute("""
        CREATE TABLE Inventory_Stock (
            item_id      VARCHAR PRIMARY KEY REFERENCES Stock (item_id),
            quantity     INTEGER NOT NULL CHECK (quantity >= 0),
            update_date  TIMESTAMP NOT NULL
        )
    """)


def load_data(con: duckdb.DuckDBPyConnection) -> None:
    """Read the CSVs and insert them into the typed tables."""
    # Stock first, so the foreign key on Inventory_Stock is satisfied.
    # TRIM() protects against stray spaces in the CSV (e.g. "Alexy G26 Pro ").
    con.execute(
        """
        INSERT INTO Stock (item_id, model, color, price, release_date)
        SELECT
            TRIM(CAST(item_id AS VARCHAR)),
            TRIM(model),
            TRIM(color),
            CAST(price AS DECIMAL(10, 2)),
            CAST(release_date AS DATE)
        FROM read_csv(?, header = true, auto_detect = true)
        """,
        [str(STOCK_CSV)],
    )

    con.execute(
        """
        INSERT INTO Inventory_Stock (item_id, quantity, update_date)
        SELECT
            TRIM(CAST(item_id AS VARCHAR)),
            CAST(quantity AS INTEGER),
            CAST(update_date AS TIMESTAMP)
        FROM read_csv(?, header = true, auto_detect = true)
        """,
        [str(INVENTORY_CSV)],
    )


def main() -> None:
    with duckdb.connect(str(DB_PATH)) as con:
        con.execute("BEGIN TRANSACTION")
        try:
            create_tables(con)
            load_data(con)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

        stock_rows = con.execute("SELECT COUNT(*) FROM Stock").fetchone()[0]
        inv_rows = con.execute("SELECT COUNT(*) FROM Inventory_Stock").fetchone()[0]
        print(f"Database ready: {DB_PATH} (Stock: {stock_rows} rows, Inventory_Stock: {inv_rows} rows)")


if __name__ == "__main__":
    main()
