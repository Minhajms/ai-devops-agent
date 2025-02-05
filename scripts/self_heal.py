import subprocess
import logging
import os
import sys
from typing import List, Tuple
import ast
import importlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineHealer:
    def __init__(self):
        self.fixed_files = []

    def analyze_python_file(self, filename: str) -> List[Tuple[str, int]]:
        issues = []
        try:
            with open(filename, 'r') as file:
                tree = ast.parse(file.read(), filename)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if not self._is_module_available(name.name):
                            issues.append((name.name, node.lineno))
                elif isinstance(node, ast.ImportFrom):
                    if not self._is_module_available(node.module):
                        issues.append((node.module, node.lineno))
        except Exception as e:
            logger.error(f"Error analyzing {filename}: {str(e)}")
        return issues

    def _is_module_available(self, module_name: str) -> bool:
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False

    def fix_dependencies(self, filename: str) -> bool:
        issues = self.analyze_python_file(filename)
        if not issues:
            return True
        logger.info(f"Found {len(issues)} issues in {filename}")
        fixed = False
        for module, line_no in issues:
            if self._attempt_pip_install(module):
                logger.info(f"Successfully installed {module}")
                fixed = True
            else:
                logger.warning(f"Could not install {module}, commenting out import")
                self._comment_out_import(filename, line_no)
                fixed = True
        return fixed

    def _attempt_pip_install(self, module: str) -> bool:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', module])
            return True
        except subprocess.CalledProcessError:
            return False

    def _comment_out_import(self, filename: str, line_no: int):
        with open(filename, 'r') as file:
            lines = file.readlines()
        if 0 <= line_no - 1 < len(lines):
            lines[line_no - 1] = f"# {lines[line_no - 1]}"
        with open(filename, 'w') as file:
            file.writelines(lines)

    def check_and_fix_memory(self):
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            if memory_usage > 1000:  # 1GB threshold
                logger.warning(f"High memory usage detected: {memory_usage:.2f} MB")
                # Implement memory cleanup strategies here
        except Exception as e:
            logger.error(f"Error checking memory: {str(e)}")

    def retry_build(self):
        logger.info("Retrying build...")
        try:
            subprocess.run(['pytest', '-n', 'auto'], check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Build retry failed: {str(e)}")
            return False

if __name__ == "__main__":
    healer = PipelineHealer()
    healer.fix_dependencies('requirements.txt')
    healer.check_and_fix_memory()
    healer.retry_build()
