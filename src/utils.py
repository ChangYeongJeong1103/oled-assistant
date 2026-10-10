import logging
import os
import time
from datetime import datetime
import config

def setup_logging():
    """Set up logging configuration."""
    if not os.path.exists(config.LOGS_DIR):
        os.makedirs(config.LOGS_DIR)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.LOGS_DIR, f"oled_assistant_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("OLED_Assistant")

logger = setup_logging()

def format_time(seconds):
    """Format seconds into readable string."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    return f"{seconds:.2f}s"


def estimate_cost_usd(model_name, prompt_tokens, completion_tokens, cached_tokens=0):
    """Estimate API cost in USD from one or more calls' token counts.

    `model_name` selects an entry in `config.MODEL_PRICING_PER_1M`.
    `prompt_tokens` includes both cached and uncached input tokens.
    `cached_tokens` are billed at the lower cached-input rate.
    `completion_tokens` contains output tokens.
    The function returns `None` when pricing for the model is unavailable.
    """
    prices = config.MODEL_PRICING_PER_1M.get(model_name)
    if prices is None:
        return None

    uncached_tokens = max(0, prompt_tokens - cached_tokens)
    cost = (
        uncached_tokens * prices["input"]
        + cached_tokens * prices["cached_input"]
        + completion_tokens * prices["output"]
    )
    return cost / 1_000_000
