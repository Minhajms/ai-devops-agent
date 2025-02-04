import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_dependency_issue():
    logger.info("Installing dependencies...")
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)

def retry_build():
    logger.info("Retrying build...")
    subprocess.run(["pytest", "-n", "auto"], check=True)

if __name__ == "__main__":
    fix_dependency_issue()
    retry_build()
