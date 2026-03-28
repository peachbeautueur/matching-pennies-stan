import os
import shutil
import sys
from pathlib import Path

import cmdstanpy
from cmdstanpy import CmdStanModel


ROOT = Path(__file__).resolve().parents[1]
STAN_PATH = ROOT / "stan" / "belief_model.stan"


def prepend_existing_paths(candidate_paths):
    path_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
    normalized_existing = {os.path.normcase(os.path.normpath(entry)) for entry in path_entries}

    additions = []
    for candidate in candidate_paths:
        candidate_str = str(candidate)
        normalized_candidate = os.path.normcase(os.path.normpath(candidate_str))
        if normalized_candidate in normalized_existing:
            continue
        if not Path(candidate_str).exists():
            continue
        additions.append(candidate_str)
        normalized_existing.add(normalized_candidate)

    if additions:
        os.environ["PATH"] = os.pathsep.join(additions + path_entries)

    return additions


def path_contains_file(filename):
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        if (Path(entry) / filename).exists():
            return True
    return False


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
    added_paths = prepend_existing_paths(common_windows_paths)

    missing = [tool for tool in required if shutil.which(tool) is None]
    if not missing:
        if added_paths:
            print("Added detected RTools paths to PATH for this session.")
        return

    print("Missing required build tools for CmdStan:", ", ".join(missing))
    print("CmdStan needs g++, mingw32-make, and cut available in PATH.")
    print("If you are on Windows, install and configure an RTools or mingw-w64 toolchain.")
    print("Then rerun the analysis script.")
    sys.exit(1)


def cmdstan_runtime_candidates():
    candidates = []

    try:
        cmdstan_path = Path(cmdstanpy.cmdstan_path())
        candidates.extend(
            [
                cmdstan_path / "bin",
                cmdstan_path / "stan" / "lib" / "stan_math" / "lib" / "tbb",
            ]
        )
    except ValueError:
        pass

    python_executable = Path(sys.executable).resolve()
    env_root = python_executable.parent
    candidates.extend(
        [
            env_root / "Library" / "ucrt64" / "bin",
            env_root / "Library" / "bin",
        ]
    )

    return candidates


def ensure_windows_runtime_paths():
    added_paths = prepend_existing_paths(cmdstan_runtime_candidates())
    if added_paths:
        print("Added detected CmdStan runtime paths to PATH for this session.")

    required_dlls = ["tbb.dll", "libstdc++-6.dll", "libgcc_s_seh-1.dll"]
    missing_dlls = [dll for dll in required_dlls if not path_contains_file(dll)]
    if missing_dlls:
        print("Missing required runtime libraries for compiled Stan models:", ", ".join(missing_dlls))
        print("Check that you are using the intended Python interpreter and conda environment.")
        print(f"Current Python: {sys.executable}")
        try:
            print(f"Current CmdStan: {cmdstanpy.cmdstan_path()}")
        except ValueError:
            print("Current CmdStan: not configured")
        sys.exit(1)


def build_model():
    if os.name == "nt":
        ensure_windows_toolchain()
        ensure_windows_runtime_paths()
    return CmdStanModel(stan_file=str(STAN_PATH))
