import re
import os 
import logging
import datetime

"""Setup logging for face swap operations"""

loggers = {}

def extract_last_percentage(log_file: str) -> float:
    if not os.path.exists(log_file):
        print("file not found:", log_file)
        return 0.0

    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    percent = 0.0
    pattern = re.compile(r'Processing:\s+(\d{1,3})%')

    for line in reversed(lines):
        match = pattern.search(line)
        if match:
            percent = float(match.group(1))
            break

    return percent

def get_logger_for_generation(OUTPUT_DIR: str, generation_id: str) -> logging.Logger:
    if generation_id in loggers:
        return loggers[generation_id]

    generation_dir = os.path.join(OUTPUT_DIR, generation_id)
    os.makedirs(generation_dir, exist_ok=True)
    log_file_path = os.path.join(generation_dir, "faceswap.log")

    logger = logging.getLogger(f"faceswap_{generation_id}")
    logger.setLevel(logging.INFO)

    # Clear existing handlers for this logger to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file_path, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    loggers[generation_id] = logger
    return logger

def log_and_print(OUTPUT_DIR: str, generation_id: str, msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(timestamp, msg)
    logger = get_logger_for_generation(OUTPUT_DIR, generation_id)
    logger.info(msg)