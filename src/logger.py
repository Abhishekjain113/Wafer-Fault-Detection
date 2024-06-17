import logging
import os
from datetime import datetime

# Create the log file name with a timestamp
LOG_FILE = f'{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.log'

# Create the path to the logs directory
logs_dir = os.path.join(os.getcwd(), 'logs')

# Ensure the logs directory exists
os.makedirs(logs_dir, exist_ok=True)

# Create the full log file path
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Set up logging configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(lineno)d %(name)s-%(levelname)s-%(message)s',
    level=logging.INFO
)

# Example usage: logging an info message
logging.info("Logging setup is complete.")
