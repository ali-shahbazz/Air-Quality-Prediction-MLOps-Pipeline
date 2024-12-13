# scripts/schedule_collection.py

import schedule
import time
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(Path('logs/data_collection.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def collect_and_version():
    """Collect data and version it with DVC"""
    try:
        logger.info("Starting scheduled data collection...")
        
        # Run data collection script
        subprocess.run(["python", "src/data/collector.py"], check=True)
        
        # Add to DVC
        subprocess.run(["dvc", "add", "data/raw"], check=True)
        
        # Commit changes
        commit_message = f"Data collection at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(["git", "add", "data/raw.dvc"], check=True)
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        
        # Push to DVC remote
        subprocess.run(["dvc", "push"], check=True)
        
        logger.info("Data collection and versioning completed successfully")
        
    except Exception as e:
        logger.error(f"Error in data collection pipeline: {str(e)}")

def main():
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    # Schedule data collection every hour
    schedule.every(1).hours.do(collect_and_version)
    
    # Run once immediately
    collect_and_version()
    
    logger.info("Data collection scheduler started...")
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()