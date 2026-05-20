# ╔══════════════════════════════════════════════════════════════╗
# ║  QuantForge — MongoDB Client & Operations                  ║
# ╚══════════════════════════════════════════════════════════════╝

import logging
from datetime import datetime, timezone, timedelta
import numpy as np
from typing import List, Dict, Optional

def clean_dict(d):
    """Recursively convert numpy types to standard python types for MongoDB insertion."""
    if isinstance(d, dict):
        return {k: clean_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [clean_dict(v) for v in d]
    elif isinstance(d, np.integer):
        return int(d)
    elif isinstance(d, np.floating):
        return float(d)
    elif isinstance(d, np.bool_):
        return bool(d)
    elif isinstance(d, np.ndarray):
        return clean_dict(d.tolist())
    else:
        return d

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

    def __init__(self, db_client):
        self.db = db_client

    def insert_strategy(self, doc: dict):
        self.db.strategies.replace_one({"strategy_id": doc["strategy_id"]}, doc, upsert=True)

    def insert_result(self, doc: dict):
        self.db.results.insert_one(doc)

    def upsert_best(self, doc: dict):
        self.db.best_strategies.replace_one({"strategy_id": doc["strategy_id"]}, doc, upsert=True)

    def save_fully_passed(self, doc: dict):
        self.db.fully_passed_strategies.replace_one({"strategy_id": doc["strategy_id"]}, doc, upsert=True)

    def insert_log(self, doc: dict):
        self.db.logs.insert_one(doc)

    def get_best(self, limit: int = 10) -> list:
        best_strats = list(self.db.best_strategies.find())
        best_strats.sort(key=lambda x: x.get("rank_score", 0), reverse=True)
        return best_strats[:limit]

    def get_logs(self, limit: int = 50) -> list:
        logs = list(self.db.logs.find())
        def sort_key(x):
            val = x.get("timestamp")
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                try:
                    return datetime.fromisoformat(val.replace("Z", "+00:00"))
                except ValueError:
                    try:
                        return datetime.strptime(val, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        return datetime.min
            return datetime.min
        logs.sort(key=sort_key, reverse=True)
        return logs[:limit]

    def get_all_results_count(self) -> int:
        return self.db.results.count_documents({})

    def get_top_strategies_with_fitness(self, limit: int = 50) -> list:
        return self.get_best(limit)

    def get_metrics_summary(self) -> dict:
        best_strats = list(self.db.best_strategies.find())
        best_strats.sort(key=lambda x: x.get("rank_score", 0), reverse=True)
        total = self.db.results.count_documents({})
        if not best_strats:
            return {"total_evaluated": total, "top_strategies": 0}
        best = best_strats[0]
        return {
            "total_evaluated": total,
            "top_strategies": len(best_strats),
            "best_score": best.get("rank_score", 0),
            "best_sharpe": best.get("metrics", {}).get("sharpe_ratio", 0),
            "best_return": best.get("metrics", {}).get("total_return_pct", 0),
            "best_avg_yearly_return": best.get("metrics", {}).get("avg_yearly_return", 0),
            "best_dd": best.get("metrics", {}).get("max_drawdown_pct", 0),
        }


import os
import json
import uuid

def _load_json(file_path):
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # convert datetime fields
            for item in data:
                for key in ["created_at", "evaluated_at", "promoted_at", "timestamp"]:
                    if key in item and isinstance(item[key], str):
                        try:
                            val = item[key].replace("Z", "+00:00")
                            item[key] = datetime.fromisoformat(val)
                        except Exception:
                            pass
            return data
    except Exception:
        return []

def _save_json(file_path, data):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, datetime):
                    return o.isoformat()
                return super().default(o)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=DateTimeEncoder, indent=2)
    except Exception as e:
        log.error(f"Failed to save local storage file {file_path}: {e}")


class InMemoryCollection:
    """Mock MongoDB collection interface for in-memory fallback with local storage persistence."""

    def __init__(self, name: str):
        self.name = name
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.file_path = os.path.join(base_dir, "cache", f"db_{name}.json")
        self.docs = _load_json(self.file_path)

    def _save(self):
        _save_json(self.file_path, self.docs)

    def find(self, filter=None, projection=None, *args, **kwargs):
        import copy
        self.docs = _load_json(self.file_path)
        copied_docs = [copy.deepcopy(d) for d in self.docs]
        
        filtered_docs = []
        if filter:
            for d in copied_docs:
                match = True
                for k, v in filter.items():
                    if k == "_id" and isinstance(v, dict) and "$in" in v:
                        if d.get(k) not in v["$in"]:
                            match = False
                            break
                    elif d.get(k) != v:
                        match = False
                        break
                if match:
                    filtered_docs.append(d)
        else:
            filtered_docs = copied_docs

        class Cursor:
            def __init__(self, docs):
                self.docs = docs
            def sort(self, key, direction=-1):
                if isinstance(key, list):
                    for k, d in key:
                        self.sort(k, d)
                    return self
                reverse = True if direction == -1 else False
                self.docs.sort(key=lambda x: x.get(key, 0) if x.get(key) is not None else 0, reverse=reverse)
                return self
            def limit(self, n):
                self.docs = self.docs[:n]
                return self
            def __iter__(self):
                return iter(self.docs)
        return Cursor(filtered_docs)

    def find_one(self, filter=None, projection=None, *args, **kwargs):
        import copy
        self.docs = _load_json(self.file_path)
        doc = self._find_one_ref(filter)
        return copy.deepcopy(doc) if doc is not None else None

    def _find_one_ref(self, filter=None):
        if not self.docs:
            return None
        if filter:
            for d in self.docs:
                match = True
                for k, v in filter.items():
                    if d.get(k) != v:
                        match = False
                        break
                if match:
                    return d
            return None
        return self.docs[0]

    def insert_one(self, doc):
        self.docs = _load_json(self.file_path)
        if "_id" not in doc:
            doc["_id"] = str(uuid.uuid4())
        self.docs.append(doc)
        self._save()
        return type('InsertOneResult', (object,), {'inserted_id': doc['_id']})()

    def update_one(self, filter, update, upsert=False):
        self.docs = _load_json(self.file_path)
        doc = self._find_one_ref(filter)
        if not doc:
            if upsert:
                new_doc = {}
                new_doc.update(filter)
                if "_id" not in new_doc:
                    new_doc["_id"] = str(uuid.uuid4())
                if "$set" in update:
                    new_doc.update(update["$set"])
                if "$inc" in update:
                    for k, v in update["$inc"].items():
                        new_doc[k] = v
                self.docs.append(new_doc)
                self._save()
            return
        if "$set" in update:
            doc.update(update["$set"])
        if "$inc" in update:
            for k, v in update["$inc"].items():
                doc[k] = doc.get(k, 0) + v
        self._save()

    def replace_one(self, filter, doc, upsert=False):
        self.docs = _load_json(self.file_path)
        existing = self._find_one_ref(filter)
        if existing:
            if "_id" in existing and "_id" not in doc:
                doc["_id"] = existing["_id"]
            self.docs.remove(existing)
        elif "_id" not in doc:
            doc["_id"] = str(uuid.uuid4())
        self.docs.append(doc)
        self._save()

    def delete_many(self, filter):
        self.docs = _load_json(self.file_path)
        if not filter:
            self.docs.clear()
            self._save()
            return
        if "_id" in filter and isinstance(filter["_id"], dict) and "$in" in filter["_id"]:
            allowed_ids = filter["_id"]["$in"]
            self.docs = [d for d in self.docs if d.get("_id") not in allowed_ids]
            self._save()

    def count_documents(self, filter):
        self.docs = _load_json(self.file_path)
        if not filter:
            return len(self.docs)
        count = 0
        for d in self.docs:
            match = True
            for k, v in filter.items():
                if d.get(k) != v:
                    match = False
                    break
            if match:
                count += 1
        return count

    def create_index(self, *args, **kwargs):
        pass


class InMemoryDBClient:
    """Mock MongoDB database object to support attribute and item access."""

    def __init__(self):
        self._collections = {}

    def __getattr__(self, name: str) -> InMemoryCollection:
        if name not in self._collections:
            self._collections[name] = InMemoryCollection(name)
        return self._collections[name]

    def __getitem__(self, name: str) -> InMemoryCollection:
        return getattr(self, name)


class MongoDB:
    """MongoDB client with all QuantForge operations."""

    def __init__(self, uri: str = MONGO_URI, db_name: str = MONGO_DB):
        if not HAS_MONGO:
            log.warning("Using local storage DB fallback")
            self.db = InMemoryDBClient()
            self._fallback = InMemoryDB(self.db)
            self._connected = False
            return

        self._fallback = None
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=1000, connectTimeoutMS=1000)
            self.client.admin.command("ping")
            self.db = self.client[db_name]
            self._connected = True
            self._setup_indexes()
            log.info(f"[OK] Connected to MongoDB: {db_name}")
        except Exception as e:
            log.warning(f"MongoDB connection failed: {e} — using local storage fallback")
            if hasattr(self, "client") and self.client:
                try:
                    self.client.close()
                except Exception:
                    pass
            self.db = InMemoryDBClient()
            self._fallback = InMemoryDB(self.db)
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
        strategy_dict = clean_dict(strategy_dict)
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
        result_dict = clean_dict(result_dict)
        if self._fallback:
            return self._fallback.insert_result(result_dict)
        result_dict["evaluated_at"] = datetime.now(timezone.utc)
        try:
            self.db.results.insert_one(result_dict)
        except Exception as e:
            log.error(f"Failed to insert result: {e}")

    def upsert_best(self, doc: dict):
        doc = clean_dict(doc)
        if self._fallback:
            return self._fallback.upsert_best(doc)
        doc["promoted_at"] = datetime.now(timezone.utc)
        try:
            self.db.best_strategies.replace_one(
                {"strategy_id": doc["strategy_id"]},
                doc, upsert=True,
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

    def save_fully_passed(self, doc: dict):
        doc = clean_dict(doc)
        if self._fallback:
            return self._fallback.save_fully_passed(doc)
        doc["passed_at"] = datetime.now(timezone.utc)
        try:
            self.db.fully_passed_strategies.replace_one(
                {"strategy_id": doc["strategy_id"]},
                doc, upsert=True,
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

    def insert_log(self, level: str, source: str, message: str, cycle: int = 0, extra: dict = None):
        doc = clean_dict({
            "timestamp": datetime.now(timezone.utc),
            "level": level,
            "source": source,
            "message": message,
            "cycle": cycle,
        })
        if extra:
            doc.update(clean_dict(extra))
        if self._fallback:
            return self._fallback.insert_log(doc)
        try:
            self.db.logs.insert_one(doc)
        except Exception:
            pass
