# ╔══════════════════════════════════════════════════════════════╗
# ║  QuantForge — Neural Surrogate Model                        ║
# ║  Predicts rank_score before full backtesting                ║
# ╚══════════════════════════════════════════════════════════════╝

import os
import joblib
import logging
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPRegressor

from backend.strategy.tree import Strategy, IndicatorNode
from backend.config import (
    SURROGATE_ENABLED, SURROGATE_MIN_SAMPLES, SURROGATE_RETRAIN_EVERY
)

log = logging.getLogger("quantforge.engine.surrogate")

MODEL_PATH = Path(os.path.dirname(__file__)) / ".." / "data" / "surrogate.joblib"


class SurrogateModel:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.stats = {
            "predictions_made": 0,
            "strategies_filtered": 0,
            "training_samples": 0,
            "mae_last_100": 0.0,
        }
        self._recent_errors = []
        
        if MODEL_PATH.exists():
            try:
                self.model = joblib.load(MODEL_PATH)
                self.is_trained = True
                log.info("Loaded pre-trained surrogate model")
            except Exception as e:
                log.warning(f"Failed to load surrogate model: {e}")

        if self.model is None:
            self.model = MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                max_iter=500,
                early_stopping=True,
                random_state=42
            )

    def _extract_features(self, strategy: Strategy) -> np.ndarray:
        """Extract ~18 features from strategy."""
        # 1-6. Risk params
        rp = strategy.risk_params
        risk_features = [
            rp.sl_atr_mult, rp.rr_ratio, rp.risk_pct,
            float(rp.cooldown), rp.trail_mult, rp.tp1_ratio
        ]
        
        # 7-10. Tree complexity
        buy_depth = strategy.buy_rule.depth() if strategy.buy_rule else 0
        sell_depth = strategy.sell_rule.depth() if strategy.sell_rule else 0
        buy_size = strategy.buy_rule.size() if strategy.buy_rule else 0
        sell_size = strategy.sell_rule.size() if strategy.sell_rule else 0
        
        # 11-16. Category counts
        nodes = []
        if strategy.buy_rule: nodes.extend(strategy.buy_rule.collect_nodes())
        if strategy.sell_rule: nodes.extend(strategy.sell_rule.collect_nodes())
        
        counts = {"ema": 0, "rsi": 0, "macd": 0, "volume": 0, "regime": 0, "multi_tf": 0}
        for n in nodes:
            if isinstance(n, IndicatorNode):
                c = n.column.lower()
                if "ema" in c: counts["ema"] += 1
                if "rsi" in c: counts["rsi"] += 1
                if "macd" in c: counts["macd"] += 1
                if "volume" in c: counts["volume"] += 1
                if "regime" in c: counts["regime"] += 1
                if "tf_4h" in c or "tf_1d" in c: counts["multi_tf"] += 1
                
        # 17. Origin encoded
        origin_map = {"random": 0, "template": 1, "crossover": 2, "mutation": 3, "bayesian": 4, "elite": 5}
        origin_val = origin_map.get(strategy.origin, 0)
        
        features = risk_features + [
            buy_depth, sell_depth, buy_size, sell_size,
            counts["ema"], counts["rsi"], counts["macd"], 
            counts["volume"], counts["regime"], counts["multi_tf"],
            float(origin_val)
        ]
        return np.array(features)

    def train(self, strategies: list, scores: list):
        if len(strategies) < SURROGATE_MIN_SAMPLES:
            log.info(f"Not enough samples to train surrogate ({len(strategies)} < {SURROGATE_MIN_SAMPLES})")
            return
            
        X = np.array([self._extract_features(s) for s in strategies])
        y = np.array(scores)
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            self.stats["training_samples"] = len(X)
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, MODEL_PATH)
            log.info(f"Surrogate model trained on {len(X)} samples")
        except Exception as e:
            log.error(f"Surrogate training failed: {e}")

    def predict(self, strategy: Strategy) -> float:
        if not self.is_trained or not SURROGATE_ENABLED:
            return 1.0 # Allow all if not active
            
        try:
            X = self._extract_features(strategy).reshape(1, -1)
            pred = self.model.predict(X)[0]
            self.stats["predictions_made"] += 1
            return float(pred)
        except Exception:
            return 1.0
            
    def record_actual(self, predicted: float, actual: float):
        error = abs(predicted - actual)
        self._recent_errors.append(error)
        if len(self._recent_errors) > 100:
            self._recent_errors.pop(0)
        self.stats["mae_last_100"] = sum(self._recent_errors) / len(self._recent_errors)
