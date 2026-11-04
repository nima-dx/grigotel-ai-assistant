"""
GrigoTel AI Assistant - Step 3: Agent tools

Four tools the agent can call:
  - search_docs(query, model=None)               -> PDF knowledge base (minsearch)
  - get_product_info(model=None)                 -> prices, colors, stock (DuckDB, read-only)
  - check_availability(model, color, quantity)   -> what can be offered (DuckDB, read-only)
  - place_order(item_id, quantity)               -> reduces stock + fake checkout link (DuckDB, WRITE)

Design notes:
  - Tools are methods on one class, so they share the DB connection and the search index.
    The docstrings and type hints are what the LLM sees as tool descriptions - keep them clear.
  - All SQL is fixed and parameterized; the LLM never writes SQL.
  - Model names are normalized ("pro", "g26 pro", "Alexy G26 Pro" -> "Alexy G26 Pro").
  - Every tool returns a plain JSON-serializable dict (no Decimal / datetime objects).
  - Each call opens its own cursor, so one shared connection is safe across Streamlit threads.

Location: src/grigotel_ai_assistant/tools.py

Usage (in the app):
    from grigotel_ai_assistant.tools import create_tools
    tools = create_tools()
"""

from datetime import datetime
from urllib.parse import urlencode

import duckdb
from minsearch import Index

try:  # works when the package is installed (uv init --package)
    from grigotel_ai_assistant.build_index import build_index, load_chunks
    from grigotel_ai_assistant.convert_docs import GENERAL
    from grigotel_ai_assistant.setup_db import DB_PATH
except ImportError:  # works when run as a script from this folder
    from build_index import build_index, load_chunks
    from convert_docs import GENERAL
    from setup_db import DB_PATH

CHECKOUT_URL = "https://shop.grigotel.example/checkout"
CURRENCY = "EUR"  # change if your CSV prices are in another currency

SPEC_RESULTS = 3      # chunks from the model's own spec sheet
GENERAL_RESULTS = 3   # chunks from manual / corporate profile / privacy policy
DEFAULT_RESULTS = 5   # when no model is given


# ── Helpers (module-level, so they are NOT exposed as tools) ─────────────

def _model_key(text: str) -> str:
    """'Alexy G26 Pro' / 'g26 pro' / 'G26-Pro' -> 'g26pro'."""
    key = text.lower().replace("alexy", "")
    return "".join(ch for ch in key if ch.isalnum())


def normalize_model(model: str, known_models: list[str]) -> str | None:
    """Map free-text model names to the exact name in the database, or None."""
    lookup = {_model_key(m): m for m in known_models}
    key = _model_key(model)
    # "pro" / "lite" / "fold" without the "g26" prefix
    return lookup.get(key) or lookup.get(f"g26{key}")


def _fetch_variants(cur: duckdb.DuckDBPyConnection, model: str | None = None) -> list[dict]:
    """All variants (model + color) with price and current stock."""
    sql = """
        SELECT s.item_id, s.model, s.color, s.price, s.release_date,
               COALESCE(i.quantity, 0) AS quantity
        FROM Stock s
        LEFT JOIN Inventory_Stock i USING (item_id)
    """
    params = []
    if model:
        sql += " WHERE s.model = ?"
        params.append(model)
    sql += " ORDER BY s.model, s.color"

    rows = cur.execute(sql, params).fetchall()
    return [
        {
            "item_id": item_id,
            "model": m,
            "color": color,
            "price": float(price),
            "release_date": str(release_date) if release_date else None,
            "quantity": quantity,
            "in_stock": quantity > 0,
        }
        for item_id, m, color, price, release_date, quantity in rows
    ]


def _money(amount: float) -> float:
    return round(amount, 2)


# ── Tools ────────────────────────────────────────────────────────────────

class GrigoTelTools:
    """The tools the GrigoTel assistant can call."""

    def __init__(self, con: duckdb.DuckDBPyConnection, index: Index):
        self.con = con
        self.index = index
        cur = con.cursor()
        self.known_models = [r[0] for r in cur.execute(
            "SELECT DISTINCT model FROM Stock ORDER BY model"
        ).fetchall()]

    def _unknown_model(self, model: str) -> dict:
        return {"error": f"Unknown model '{model}'.", "valid_models": self.known_models}

    # 1 ── Knowledge base ─────────────────────────────────────────────────
    def search_docs(self, query: str, model: str | None = None) -> dict:
        """
        Search GrigoTel's documents: product specification sheets, the user manual
        (setup, maintenance, battery management, troubleshooting), the corporate
        profile and the privacy & compliance policy.

        Use for technical specifications, how-to questions, company information
        and privacy questions. Do NOT use for prices, colors or stock.

        Args:
            query: The search query, e.g. "battery capacity" or "how to replace the battery".
            model: Optional phone model, e.g. "Alexy G26 Pro". When given, results
                include that model's specification sheet plus general documents.
                When the spec sheet and the user manual disagree, the spec sheet is correct.
        """
        if model:
            exact = normalize_model(model, self.known_models)
            if exact is None:
                return self._unknown_model(model)
            results = self.index.search(query, filter_dict={"model": exact}, num_results=SPEC_RESULTS)
            results += self.index.search(query, filter_dict={"model": GENERAL}, num_results=GENERAL_RESULTS)
        else:
            results = self.index.search(query, num_results=DEFAULT_RESULTS)

        return {
            "results": [
                {
                    "source": r["filename"],
                    "doc_type": r["doc_type"],
                    "model": r["model"],
                    "content": r["content"],
                }
                for r in results
            ]
        }

    # 2 ── Catalog ────────────────────────────────────────────────────────
    def get_product_info(self, model: str | None = None) -> dict:
        """
        Get prices, available colors and stock for GrigoTel phones from the database.
        This is the ONLY source for prices and availability.

        Args:
            model: Optional phone model, e.g. "Alexy G26 Pro". Leave empty to list all models.
        """
        cur = self.con.cursor()
        if model:
            exact = normalize_model(model, self.known_models)
            if exact is None:
                return self._unknown_model(model)
            variants = _fetch_variants(cur, exact)
        else:
            variants = _fetch_variants(cur)

        products: dict[str, list[dict]] = {}
        for v in variants:
            products.setdefault(v["model"], []).append(
                {k: v[k] for k in ("item_id", "color", "price", "quantity", "in_stock")}
            )
        return {
            "currency": CURRENCY,
            "products": [{"model": m, "variants": vs} for m, vs in products.items()],
        }

    # 3 ── Availability check (read-only) ─────────────────────────────────
    def check_availability(self, model: str, color: str, quantity: int) -> dict:
        """
        Check whether a quantity of a specific model and color can be ordered.
        Read-only: does NOT place an order. Always call this before place_order.

        Returns a status:
          - "available":        the full quantity can be ordered (item_id and total price included)
          - "partial":          fewer units in stock than requested; offer available_quantity
          - "out_of_stock":     this color is sold out; offer the alternative colors
          - "color_not_offered": this model doesn't come in that color; list available colors

        Args:
            model: Phone model, e.g. "Alexy G26 Pro".
            color: Color name, e.g. "Black".
            quantity: Number of phones the customer wants (1 or more).
        """
        if quantity < 1:
            return {"error": "Quantity must be at least 1."}

        exact = normalize_model(model, self.known_models)
        if exact is None:
            return self._unknown_model(model)

        variants = _fetch_variants(self.con.cursor(), exact)
        match = next((v for v in variants if v["color"].lower() == color.strip().lower()), None)

        # Other colors of the same model that are in stock
        alternatives = [
            {"item_id": v["item_id"], "color": v["color"], "price": v["price"], "quantity": v["quantity"]}
            for v in variants
            if v["in_stock"] and (match is None or v["item_id"] != match["item_id"])
        ]

        if match is None:
            return {
                "status": "color_not_offered",
                "model": exact,
                "requested_color": color,
                "available_colors": alternatives,
            }

        result = {
            "model": exact,
            "color": match["color"],
            "item_id": match["item_id"],
            "unit_price": match["price"],
            "requested_quantity": quantity,
            "available_quantity": match["quantity"],
            "currency": CURRENCY,
        }

        if match["quantity"] >= quantity:
            return {**result, "status": "available", "total_price": _money(match["price"] * quantity)}

        if match["quantity"] > 0:
            return {
                **result,
                "status": "partial",
                "total_price_for_available": _money(match["price"] * match["quantity"]),
                # colors that can cover the FULL requested quantity
                "alternative_colors": [a for a in alternatives if a["quantity"] >= quantity],
            }

        return {**result, "status": "out_of_stock", "alternative_colors": alternatives}

    # 4 ── Order (WRITES to the database) ─────────────────────────────────
    def place_order(self, item_id: str, quantity: int) -> dict:
        """
        Place an order: reduces stock and returns a checkout link.
        ONLY call this after the customer has explicitly confirmed the item
        (model + color), the quantity and the total price.

        Args:
            item_id: The item_id returned by check_availability.
            quantity: Number of phones to order (1 or more).
        """
        if quantity < 1:
            return {"error": "Quantity must be at least 1."}

        cur = self.con.cursor()

        # One atomic statement: only succeeds if there is still enough stock.
        # This re-checks stock at write time, in case it changed since check_availability.
        row = cur.execute(
            """
            UPDATE Inventory_Stock
            SET quantity = quantity - ?, update_date = ?
            WHERE item_id = ? AND quantity >= ?
            RETURNING quantity
            """,
            [quantity, datetime.now(), item_id, quantity],
        ).fetchone()

        if row is None:
            current = cur.execute(
                "SELECT quantity FROM Inventory_Stock WHERE item_id = ?", [item_id]
            ).fetchone()
            if current is None:
                return {"status": "error", "error": f"Unknown item_id '{item_id}'."}
            return {
                "status": "insufficient_stock",
                "item_id": item_id,
                "requested_quantity": quantity,
                "available_quantity": current[0],
            }

        model, color, price = cur.execute(
            "SELECT model, color, price FROM Stock WHERE item_id = ?", [item_id]
        ).fetchone()

        return {
            "status": "confirmed",
            "item_id": item_id,
            "model": model,
            "color": color,
            "quantity": quantity,
            "unit_price": float(price),
            "total_price": _money(float(price) * quantity),
            "currency": CURRENCY,
            "checkout_url": f"{CHECKOUT_URL}?{urlencode({'item': item_id, 'qty': quantity})}",
        }


def create_tools(db_path=DB_PATH) -> GrigoTelTools:
    """Build the tools with a DB connection and the search index (used by the app)."""
    con = duckdb.connect(str(db_path))
    index = build_index(load_chunks())
    return GrigoTelTools(con, index)
