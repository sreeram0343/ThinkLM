import os
import sys
import uuid
import logging
import subprocess
from typing import Tuple

logger = logging.getLogger("ThinkLM.Sandbox")

class TimeoutExpired(Exception):
    """
    Exception raised when a sandboxed execution times out.
    Carries the stdout and stderr produced up to the timeout.
    """
    def __init__(self, stdout: str, stderr: str, timeout: float):
        super().__init__(f"Execution timed out after {timeout} seconds")
        self.stdout = stdout
        self.stderr = stderr
        self.timeout = timeout

def execute_sandboxed_code(script_code: str, timeout_sec: int = 5) -> Tuple[int, str, str]:
    """
    Executes raw Python code in a secure, isolated subprocess restricted to the ./sandbox/ folder.
    
    Args:
        script_code (str): The code content to execute.
        timeout_sec (int): The execution timeout in seconds.
        
    Returns:
        Tuple[int, str, str]: (return_code, stdout, stderr)
        
    Raises:
        TimeoutExpired: If the execution exceeds the timeout limit.
    """
    # Define sandboxing folder path
    sandbox_dir = os.path.abspath("./sandbox")
    os.makedirs(sandbox_dir, exist_ok=True)
    
    # Generate unique filename for temporary execution script
    file_id = uuid.uuid4().hex
    temp_filename = f"temp_run_{file_id}.py"
    temp_filepath = os.path.join(sandbox_dir, temp_filename)
    
    logger.info(f"Writing script code to sandbox path: {temp_filepath}")
    with open(temp_filepath, "w", encoding="utf-8") as f:
        f.write(script_code)
        
    # Execute the command with cwd set to the sandbox directory
    # On Windows, using sys.executable ensures the same Python environment is used
    try:
        process = subprocess.Popen(
            [sys.executable, temp_filename],
            cwd=sandbox_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8"
        )
        
        try:
            stdout, stderr = process.communicate(timeout=timeout_sec)
            return process.returncode, stdout, stderr
        except subprocess.TimeoutExpired:
            logger.warning(f"Process exceeded timeout of {timeout_sec}s. Escalating termination...")
            # Windows/Unix compatible escalation
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
            
            raise TimeoutExpired(stdout=stdout or "", stderr=stderr or "", timeout=timeout_sec)
            
    finally:
        # Secure file cleaning on completion
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
                logger.info(f"Successfully cleaned up temporary sandbox file: {temp_filename}")
            except Exception as e:
                logger.error(f"Failed to remove sandbox temporary file: {e}")
