from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    code_dir = repo_root / "code"
    script_path = code_dir / "scripts" / "test_retrieval.py"

    os.chdir(code_dir)
    sys.argv[0] = str(script_path)
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
