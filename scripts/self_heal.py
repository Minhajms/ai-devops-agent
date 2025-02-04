import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_dependency_issue(module_name):
    logger.info(f"Attempting to install missing dependency: {module_name}")
    subprocess.run(["pip", "install", module_name], check=True)

def retry_build():
    logger.info("Retrying build...")
    subprocess.run(["pytest", "-n", "auto"], check=True)

if __name__ == "__main__":
    missing_module = "nonexistent_module"  # Change this dynamically if needed
    fix_dependency_issue(missing_module)
    retry_build()

