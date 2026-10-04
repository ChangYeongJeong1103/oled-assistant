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
    """
    Estimate the API cost (USD) of one or more calls from their token counts.

    Args:
        model_name: Model name used as the key in config.MODEL_PRICING_PER_1M.
        prompt_tokens: Total input tokens, including the cached ones.
        completion_tokens: Output tokens.
        cached_tokens: Input tokens served from the prompt cache. These are
            billed at the cheaper cached-input rate.

    Returns:
        float: Estimated cost in USD, or None if the model has no entry in
        config.MODEL_PRICING_PER_1M.
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
