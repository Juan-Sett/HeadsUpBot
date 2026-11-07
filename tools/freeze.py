"""Freeze the current bot (agent.py + bot/) as a practice opponent.

    python tools/freeze.py v1          # -> opponents/frozen/v1/
    python sim.py agent v1             # play the current bot against it
"""

import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    dest = os.path.join(ROOT, "opponents", "frozen", sys.argv[1])
    if os.path.exists(dest):
        print(f"{dest} already exists; pick a new name")
        return 1
    os.makedirs(dest)
    shutil.copy2(os.path.join(ROOT, "agent.py"), dest)
    shutil.copytree(os.path.join(ROOT, "bot"), os.path.join(dest, "bot"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    print(f"froze the current bot into {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
