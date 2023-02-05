"""
Pytest reads this file in before doing any work. Read more about conftest.py under:
  - https://docs.pytest.org/en/stable/fixture.html
  - https://docs.pytest.org/en/stable/writing_plugins.html
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.absolute()))
sys.path.append(str(Path(__file__).parent.parent.absolute() / "src"))

from utils.fixtures import client  # noqa: F401, E402
