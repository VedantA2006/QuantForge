# ╔══════════════════════════════════════════════════════════════╗
# ║  QuantForge — Shared Utilities                              ║
# ╚══════════════════════════════════════════════════════════════╝

import numpy as np


def clean_for_mongo(d):
    """Recursively convert numpy types to standard Python types for MongoDB/JSON."""
    if isinstance(d, dict):
        return {k: clean_for_mongo(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [clean_for_mongo(v) for v in d]
    elif isinstance(d, tuple):
        return [clean_for_mongo(v) for v in d]
    elif isinstance(d, np.bool_):
        return bool(d)
    elif isinstance(d, np.integer):
        return int(d)
    elif isinstance(d, np.floating):
        return float(d)
    elif isinstance(d, np.ndarray):
        return clean_for_mongo(d.tolist())
    elif isinstance(d, (np.generic,)):
        return d.item()
    else:
        return d
