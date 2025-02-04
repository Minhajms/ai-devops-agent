import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_module_available(module_name):
    """Check if a module exists on PyPI before trying to install it."""
    result = subprocess.run(["pip", "index", "versions", module_name], capture_output=True, text=True)
    return module_name in result.stdout

def fix_dependency_issue(module_name):
    if is_module_available(module_name):
        logger.info(f"Attempting to install missing dependency: {module_name}")
        try:
            subprocess.run(["pip", "install", module_name], check=True)
        except subprocess.CalledProcessError:
            logger.error(f"Failed to install {module_name}. It might not be available.")
    else:
        logger.error(f"Module {module_name} is not available in PyPI. Skipping installation.")

def retry_build():
    logger.info("Retrying build...")
    subprocess.run(["pytest", "-n", "auto"], check=True)

if __name__ == "__main__":
    missing_module = "nonexistent_module"  # Change this dynamically if needed
    fix_dependency_issue(missing_module)
    retry_build()

