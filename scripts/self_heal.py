import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def comment_out_missing_imports(filename, missing_module):
    with open(filename, "r") as file:
        lines = file.readlines()

    with open(filename, "w") as file:
        inside_block = False  # Track if we're inside a function/class
        for i, line in enumerate(lines):
            stripped = line.strip()

            # Detect the start of a function or class
            if stripped.startswith(("def ", "class ")):
                inside_block = True
            
            # Ensure we handle indentation correctly
            if f"import {missing_module}" in stripped:
                leading_spaces = len(line) - len(line.lstrip())  # Preserve indentation
                lines[i] = f"{' ' * leading_spaces}# {line.lstrip()}"
                logger.info(f"Commented out missing import: {line.strip()}")

                # If inside a function/class, add `pass` to prevent indentation errors
                if inside_block:
                    lines.insert(i + 1, f"{' ' * leading_spaces}pass  # Added to prevent IndentationError\n")

            # Reset when we leave a block
            elif stripped and not stripped.startswith("#") and not stripped.startswith(("def ", "class ")):
                inside_block = False  

    with open(filename, "w") as file:
        file.writelines(lines)

def retry_build():
    logger.info("Retrying build...")
    subprocess.run(["pytest", "-n", "auto"], check=True)

if __name__ == "__main__":
    missing_module = "nonexistent_module"
    comment_out_missing_imports("scripts/test_sample.py", missing_module)
    retry_build()

