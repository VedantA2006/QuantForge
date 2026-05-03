# ╔══════════════════════════════════════════════════════════════╗
# ║  QuantForge — MongoDB Client & Operations                  ║
# ╚══════════════════════════════════════════════════════════════╝

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

from backend.config import MONGO_URI, MONGO_DB, LOG_TTL_DAYS, TOP_N_BEST

log = logging.getLogger("quantforge.db.mongo")

try:
    from pymongo import MongoClient, DESCENDING
    from pymongo.errors import ConnectionFailure
    HAS_MONGO = True
except ImportError:
    HAS_MONGO = False
    log.warning("pymongo not installed — using in-memory fallback")


class InMemoryDB:
    """Fallback when MongoDB is unavailable."""

    def __init__(self):
        self.strategies = []
        self.results = []
        self.best_strategies = []
        self.fully_passed_strategies = []
        self.logs = []

    def insert_strategy(self, doc: dict):
        self.strategies.append(doc)

    def insert_result(self, doc: dict):
        self.results.append(doc)
        if len(self.results) > 1000:
            self.results = self.results[-500:]

    def upsert_best(self, doc: dict):
        existing = [i for i, b in enumerate(self.best_strategies)
                     if b.get("strategy_id") == doc.get("strategy_id")]
        if existing:
            self.best_strategies[existing[0]] = doc
        else:
            self.best_strategies.append(doc)
        self.best_strategies.sort(key=lambda x: x.get("rank_score", 0), reverse=True)
        self.best_strategies = self.best_strategies[:TOP_N_BEST]

    def save_fully_passed(self, doc: dict):
        existing = [i for i, b in enumerate(self.fully_passed_strategies)
                     if b.get("strategy_id") == doc.get("strategy_id")]
        if existing:
            self.fully_passed_strategies[existing[0]] = doc
        else:
            self.fully_passed_strategies.append(doc)

    def insert_log(self, doc: dict):
        self.logs.append(doc)
        if len(self.logs) > 500:
            self.logs = self.logs[-250:]

    def get_best(self, limit: int = 10) -> list:
        return self.best_strategies[:limit]

    def get_logs(self, limit: int = 50) -> list:
        return list(reversed(self.logs[-limit:]))

    def get_all_results_count(self) -> int:
        return len(self.results)

    def get_top_strategies_with_fitness(self, limit: int = 50) -> list:
        return self.best_strategies[:limit]

    def get_metrics_summary(self) -> dict:
        if not self.best_strategies:
            return {"total_evaluated": len(self.results), "top_strategies": 0}
        best = self.best_strategies[0] if self.best_strategies else {}
        return {
            "total_evaluated": len(self.results),
            "top_strategies": len(self.best_strategies),
            "best_score": best.get("rank_score", 0),
            "best_sharpe": best.get("metrics", {}).get("sharpe_ratio", 0),
            "best_return": best.get("metrics", {}).get("total_return_pct", 0),
            "best_avg_yearly_return": best.get("metrics", {}).get("avg_yearly_return", 0),
            "best_dd": best.get("metrics", {}).get("max_drawdown_pct", 0),
        }


class MongoDB:
    """MongoDB client with all QuantForge operations."""

    def __init__(self, uri: str = MONGO_URI, db_name: str = MONGO_DB):
        if not HAS_MONGO:
            log.warning("Using in-memory DB fallback")
            self._fallback = InMemoryDB()
            self._connected = False
            return

        self._fallback = None
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command("ping")
            self.db = self.client[db_name]
            self._connected = True
            self._setup_indexes()
            log.info(f"[OK] Connected to MongoDB: {db_name}")
        except Exception as e:
            log.warning(f"MongoDB connection failed: {e} — using in-memory fallback")
            self._fallback = InMemoryDB()
            self._connected = False

    def _setup_indexes(self):
        if not self._connected:
            return
        self.db.strategies.create_index("strategy_id", unique=True)
        self.db.results.create_index("strategy_id")
        self.db.results.create_index([("rank_score", DESCENDING)])
        self.db.best_strategies.create_index("strategy_id", unique=True)
        self.db.best_strategies.create_index([("rank_score", DESCENDING)])
        self.db.fully_passed_strategies.create_index("strategy_id", unique=True)
        self.db.logs.create_index("timestamp")
        # TTL index for auto-cleanup of old logs
        try:
            self.db.logs.create_index(
                "timestamp",
                expireAfterSeconds=LOG_TTL_DAYS * 86400,
                name="log_ttl",
            )
        except Exception:
            pass

    # ── Strategy CRUD ─────────────────────────────────────────────

    def insert_strategy(self, strategy_dict: dict):
        if self._fallback:
            return self._fallback.insert_strategy(strategy_dict)
        strategy_dict["created_at"] = datetime.now(timezone.utc)
        try:
            self.db.strategies.replace_one(
                {"strategy_id": strategy_dict["strategy_id"]},
                strategy_dict, upsert=True,
            )
        except Exception as e:
            log.error(f"Failed to insert strategy: {e}")

    def insert_result(self, result_dict: dict):
        if self._fallback:
            return self._fallback.insert_result(result_dict)
        result_dict["evaluated_at"] = datetime.now(timezone.utc)
        try:
            self.db.results.insert_one(result_dict)
        except Exception as e:
            log.error(f"Failed to insert result: {e}")

    def upsert_best(self, best_dict: dict):
        if self._fallback:
            return self._fallback.upsert_best(best_dict)
        best_dict["promoted_at"] = datetime.now(timezone.utc)
        try:
            self.db.best_strategies.replace_one(
                {"strategy_id": best_dict["strategy_id"]},
                best_dict, upsert=True,
            )
            # Keep only top N
            count = self.db.best_strategies.count_documents({})
            if count > TOP_N_BEST:
                worst = list(self.db.best_strategies.find().sort(
                    "rank_score", 1).limit(count - TOP_N_BEST))
                ids = [w["_id"] for w in worst]
                self.db.best_strategies.delete_many({"_id": {"$in": ids}})
        except Exception as e:
            log.error(f"Failed to upsert best strategy: {e}")

    def save_fully_passed(self, passed_dict: dict):
        if self._fallback:
            return self._fallback.save_fully_passed(passed_dict)
        passed_dict["passed_at"] = datetime.now(timezone.utc)
        try:
            self.db.fully_passed_strategies.replace_one(
                {"strategy_id": passed_dict["strategy_id"]},
                passed_dict, upsert=True,
            )
        except Exception as e:
            log.error(f"Failed to save fully passed strategy: {e}")

    # ── Queries ───────────────────────────────────────────────────

    def get_best(self, limit: int = 10) -> list:
        if self._fallback:
            return self._fallback.get_best(limit)
        try:
            cursor = self.db.best_strategies.find(
                {}, {"_id": 0}
            ).sort("rank_score", DESCENDING).limit(limit)
            return list(cursor)
        except Exception:
            return []

    def get_logs(self, limit: int = 50) -> list:
        if self._fallback:
            return self._fallback.get_logs(limit)
        try:
            cursor = self.db.logs.find(
                {}, {"_id": 0}
            ).sort("timestamp", DESCENDING).limit(limit)
            return list(cursor)
        except Exception:
            return []

    def get_all_results_count(self) -> int:
        if self._fallback:
            return self._fallback.get_all_results_count()
        try:
            return self.db.results.count_documents({})
        except Exception:
            return 0

    def get_top_strategies_with_fitness(self, limit: int = 50) -> list:
        if self._fallback:
            return self._fallback.get_top_strategies_with_fitness(limit)
        try:
            cursor = self.db.best_strategies.find(
                {}, {"_id": 0}
            ).sort("rank_score", DESCENDING).limit(limit)
            return list(cursor)
        except Exception:
            return []

    def get_metrics_summary(self) -> dict:
        if self._fallback:
            return self._fallback.get_metrics_summary()
        try:
            total = self.db.results.count_documents({})
            n_best = self.db.best_strategies.count_documents({})
            best = self.db.best_strategies.find_one(
                {}, {"_id": 0}, sort=[("rank_score", DESCENDING)]
            )
            return {
                "total_evaluated": total,
                "top_strategies": n_best,
                "best_score": best.get("rank_score", 0) if best else 0,
                "best_sharpe": best.get("metrics", {}).get("sharpe_ratio", 0) if best else 0,
                "best_return": best.get("metrics", {}).get("total_return_pct", 0) if best else 0,
                "best_avg_yearly_return": best.get("metrics", {}).get("avg_yearly_return", 0) if best else 0,
                "best_dd": best.get("metrics", {}).get("max_drawdown_pct", 0) if best else 0,
            }
        except Exception:
            return {"total_evaluated": 0, "top_strategies": 0}

    # ── Logging ───────────────────────────────────────────────────

    def insert_log(self, level: str, module: str, message: str,
                   cycle: int = 0, extra: dict = None):
        doc = {
            "timestamp": datetime.now(timezone.utc),
            "level": level,
            "module": module,
            "message": message,
            "cycle": cycle,
        }
        if extra:
            doc.update(extra)
        if self._fallback:
            return self._fallback.insert_log(doc)
        try:
            self.db.logs.insert_one(doc)
        except Exception:
            pass
