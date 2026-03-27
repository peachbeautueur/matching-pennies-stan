import os
import shutil
import sys
from pathlib import Path

from cmdstanpy import CmdStanModel


ROOT = Path(__file__).resolve().parents[1]
STAN_PATH = ROOT / "stan" / "belief_model.stan"


def ensure_windows_toolchain():
    required = ["g++", "mingw32-make", "cut"]
    missing = [tool for tool in required if shutil.which(tool) is None]

    if not missing:
        return

    common_windows_paths = [
        r"C:\rtools45\usr\bin",
        r"C:\rtools45\x86_64-w64-mingw32.static.posix\bin",
        r"C:\rtools44\usr\bin",
        r"C:\rtools44\x86_64-w64-mingw32.static.posix\bin",
        r"C:\rtools43\usr\bin",
        r"C:\rtools43\x86_64-w64-mingw32.static.posix\bin",
    ]
    existing_paths = [path for path in common_windows_paths if Path(path).exists()]

    if existing_paths:
        os.environ["PATH"] = os.pathsep.join(existing_paths + [os.environ.get("PATH", "")])
        missing = [tool for tool in required if shutil.which(tool) is None]
        if not missing:
            print("Added detected RTools paths to PATH for this session.")
            return

    print("Missing required build tools for CmdStan:", ", ".join(missing))
    print("CmdStan needs g++, mingw32-make, and cut available in PATH.")
    print("If you are on Windows, install and configure an RTools or mingw-w64 toolchain.")
    if existing_paths:
        print("Detected candidate toolchain paths on this machine:")
        for path in existing_paths:
            print(f"  {path}")
    print("Then rerun the analysis script.")
    sys.exit(1)


def build_model():
    ensure_windows_toolchain()
    return CmdStanModel(stan_file=str(STAN_PATH))
