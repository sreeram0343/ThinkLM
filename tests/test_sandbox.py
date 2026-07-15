import os
import pytest
from src.utils.sandbox import execute_sandboxed_code, TimeoutExpired

def test_sandbox_success():
    code = "print('Hello from the sandbox!')"
    rc, stdout, stderr = execute_sandboxed_code(code)
    assert rc == 0
    assert "Hello from the sandbox!" in stdout
    assert stderr == ""

def test_sandbox_error():
    code = "raise ValueError('An intentional error')"
    rc, stdout, stderr = execute_sandboxed_code(code)
    assert rc != 0
    assert "ValueError: An intentional error" in stderr

def test_sandbox_timeout():
    # Attempting to sleep for 10 seconds with a 2-second timeout
    code = "import time\ntime.sleep(10)"
    with pytest.raises(TimeoutExpired) as exc_info:
        execute_sandboxed_code(code, timeout_sec=2)
    
    assert exc_info.value.timeout == 2
    # Ensure stdout/stderr attributes are present (even if empty)
    assert isinstance(exc_info.value.stdout, str)
    assert isinstance(exc_info.value.stderr, str)

def test_sandbox_cleanup():
    # Assert sandbox temp files are removed.
    # We can check that the files are deleted on success/failure.
    sandbox_dir = os.path.abspath("./sandbox")
    if os.path.exists(sandbox_dir):
        initial_files = set(os.listdir(sandbox_dir))
    else:
        initial_files = set()
        
    code = "print('checking files')"
    execute_sandboxed_code(code)
    
    if os.path.exists(sandbox_dir):
        post_files = set(os.listdir(sandbox_dir))
    else:
        post_files = set()
        
    assert initial_files == post_files
