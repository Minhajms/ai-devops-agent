import pytest
import nonexistent_module 
def test_example():
    assert 1 == 1

def test_dependency_issue():
    import nonexistent_module  # This will raise ModuleNotFoundError
