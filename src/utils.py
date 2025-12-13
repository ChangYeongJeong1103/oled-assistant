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
