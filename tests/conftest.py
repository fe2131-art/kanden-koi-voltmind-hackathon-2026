"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add project root and src/ to PYTHONPATH
# - project_root: allows `from src.safety_agent.xxx import ...`
# - src_path: allows `from safety_agent.xxx import ...`
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(1, str(src_path))
