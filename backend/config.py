# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  QuantForge — Configuration                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os
from dotenv import load_dotenv

load_dotenv()

# ── Bybit Data ────────────────────────────────────────────────────────────────
BYBIT_BASE_URL = "https://api.bybit.com/v5/market/kline"
SYMBOL = os.getenv("QF_SYMBOL", "BTCUSDT")
INTERVALS = os.getenv("QF_INTERVALS", "1h,4h").split(",")
PRIMARY_INTERVAL = INTERVALS[0]
DURATION_YEARS = int(os.getenv("QF_DURATION_YEARS", "5"))

# ── Data Cache ────────────────────────────────────────────────────────────────
CACHE_DIR = os.getenv("QF_CACHE_DIR", os.path.join(os.path.dirname(__file__), "cache"))
CACHE_TTL_HOURS = int(os.getenv("QF_CACHE_TTL_HOURS", "1"))

# ── Strategy Generation ──────────────────────────────────────────────────────
BATCH_SIZE = int(os.getenv("QF_BATCH_SIZE", "20"))
MAX_TREE_DEPTH = int(os.getenv("QF_MAX_TREE_DEPTH", "4"))
RANDOM_RATIO = 0.20        # 20% random strategies per batch
TEMPLATE_RATIO = 0.50      # 50% template-based
GA_RATIO = 0.30             # 30% GA offspring

# ── Backtesting ───────────────────────────────────────────────────────────────
INITIAL_BALANCE = float(os.getenv("QF_INITIAL_BALANCE", "10000"))
DEFAULT_RISK_PCT = 0.01
DEFAULT_FEE = 0.0005       # Bybit taker fee
DEFAULT_SLIPPAGE = 0.0001
DEFAULT_SPREAD = 0.0002
COOLDOWN_BARS = int(os.getenv("QF_COOLDOWN_BARS", "3"))

# ── Genetic Algorithm ────────────────────────────────────────────────────────
GA_POPULATION_SIZE = int(os.getenv("QF_GA_POP_SIZE", "100"))
GA_GENERATIONS = int(os.getenv("QF_GA_GENERATIONS", "30"))
GA_TOURNAMENT_K = 3
GA_CROSSOVER_RATE = 0.7
GA_MUTATION_RATE = 0.15
GA_PARAM_MUTATION_RATE = 0.2
GA_ELITISM_COUNT = 10
GA_EVOLVE_EVERY_N_CYCLES = int(os.getenv("QF_GA_EVOLVE_EVERY", "3"))

# ── Validation ────────────────────────────────────────────────────────────────
VALIDATION_FOLDS = 5
TRAIN_RATIO = 0.7
MIN_TRADES_PER_FOLD = 15
OOS_SHARPE_DECAY_LIMIT = 0.50    # reject if OOS sharpe < 50% of IS
OOS_DD_AMPLIFY_LIMIT = 2.0       # reject if OOS DD > 2x IS DD
WR_VARIANCE_LIMIT = 0.15         # reject if WR variance across folds > 15%

# ── Ranking Weights ───────────────────────────────────────────────────────────
RANK_W_SHARPE = 0.30
RANK_W_RETURN = 0.20
RANK_W_DRAWDOWN = 0.20
RANK_W_CONSISTENCY = 0.30

# ── MongoDB ───────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("QF_MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("QF_MONGO_DB", "quantforge")
LOG_TTL_DAYS = int(os.getenv("QF_LOG_TTL_DAYS", "7"))
TOP_N_BEST = int(os.getenv("QF_TOP_N_BEST", "25"))

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("QF_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("QF_API_PORT", "8000"))
CORS_ORIGINS = os.getenv("QF_CORS_ORIGINS", "http://localhost:3000,*").split(",")

# ── Discovery Loop ────────────────────────────────────────────────────────────
LOOP_DELAY_SECONDS = int(os.getenv("QF_LOOP_DELAY", "5"))
MAX_STRATEGIES_STORED = int(os.getenv("QF_MAX_STORED", "500"))

# ── ML Upgrades ───────────────────────────────────────────────────────────────
SURROGATE_ENABLED = os.getenv("QF_SURROGATE", "true").lower() == "true"
SURROGATE_FILTER_THRESHOLD = float(os.getenv("QF_SURROGATE_THRESHOLD", "0.5"))
SURROGATE_MIN_SAMPLES = int(os.getenv("QF_SURROGATE_MIN_SAMPLES", "200"))
SURROGATE_RETRAIN_EVERY = int(os.getenv("QF_SURROGATE_RETRAIN", "10"))
RL_ENABLED = os.getenv("QF_RL", "true").lower() == "true"
RL_BATCH_RATIO = float(os.getenv("QF_RL_RATIO", "0.20"))
RL_LR = float(os.getenv("QF_RL_LR", "0.01"))
POOL_ENABLED = os.getenv("QF_POOL", "true").lower() == "true"
HOF_ENABLED = os.getenv("QF_HOF", "true").lower() == "true"
PRUNER_ENABLED = os.getenv("QF_PRUNER", "true").lower() == "true"
PRUNER_MIN_COMPLEXITY = int(os.getenv("QF_PRUNER_MIN_COMPLEXITY", "4"))
