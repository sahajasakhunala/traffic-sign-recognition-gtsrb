import os

# Resolve project root dynamically
# C:\Users\LENOVO\OneDrive\Documents\traffic-sign-recognition-gtsrb\src\utils\paths.py -> root is 3 levels up
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(UTILS_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

def get_project_root() -> str:
    """Returns the absolute path to the project root directory."""
    return PROJECT_ROOT

def get_absolute_path(relative_path: str) -> str:
    """Resolves a path relative to the project root to an absolute path.
    
    If the path is already absolute, it is returned unchanged.
    """
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.abspath(os.path.join(PROJECT_ROOT, relative_path))
