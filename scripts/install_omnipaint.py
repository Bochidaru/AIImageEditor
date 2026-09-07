"""Install the official OmniPaint runtime source into third_party/OmniPaint."""

from __future__ import annotations

import subprocess
from pathlib import Path


root = Path(__file__).resolve().parents[1]
destination = root / "third_party" / "OmniPaint"
if destination.exists():
    raise SystemExit(f"Already exists: {destination}")
destination.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(
    ["git", "clone", "--depth", "1", "https://github.com/yeates/OmniPaint.git", str(destination)],
    check=True,
)
print(f"Installed official OmniPaint runtime at {destination}")
