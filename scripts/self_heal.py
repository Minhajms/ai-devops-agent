import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_dependency_issue():
    logger.info("Fixing dependency issue...")
    subprocess.run(["pip", "install", "-r", "requirements.txt"])

def retry_build():
    logger.info("Retrying build...")
    subprocess.run(["python", "-m", "pytest"])
