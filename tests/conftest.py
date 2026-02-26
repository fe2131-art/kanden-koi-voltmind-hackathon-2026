"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add src/ to PYTHONPATH so that 'run' and 'safety_agent' can be imported directly
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
