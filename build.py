#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: shihua
# Designer: shihua
# Coder: shihua
# Email: 15021408795@163.com
# License: Apache-2.0
# Copyright (c) 2026 shihua. All rights reserved.

'''
Module Introduction
-------------------
Build SeaWave with uv and pyproject.toml, install the newest wheel on request,
and verify the installed package with every runtime dependency.

Dependency closure
------------------
The script checks llvm-openmp in the active Conda environment and installs it
from conda-forge when missing. The wheel installer resolves JAX, JAXLIB,
LightGBM, and NumPy from pyproject.toml. Post-install verification imports all
of them outside the source directory. uv is installed with pip when absent.

Usage examples
--------------
.. code-block:: bash

    conda activate seawave
    python build.py --clean --install
    python build.py --verbose

Precautions
-----------
An active Conda environment is required because llvm-openmp is a Conda runtime
dependency rather than a pip dependency. Python dependencies are installed only
with --install; the default build-only command still checks build prerequisites.
'''

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"


def read_version() -> str:
    '''Read the package version from the single pyproject.toml declaration.'''
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise RuntimeError("No project version found in pyproject.toml")
    return match.group(1)


def run(command: list[str], cwd: Path | None = None, verbose: bool = False) -> None:
    '''Run a command and stop immediately with captured diagnostics on failure.'''
    if verbose:
        print(f"  $ {' '.join(str(item) for item in command)}")
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=not verbose,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"  Command failed: {' '.join(str(item) for item in command)}")
        if not verbose and result.stdout:
            print(result.stdout)
        if not verbose and result.stderr:
            print(result.stderr)
        raise SystemExit(result.returncode)


def conda_openmp_packages(conda_executable: str) -> list[dict]:
    '''Return package records, treating Conda's no-match response as an empty list.'''
    result = subprocess.run(
        [conda_executable, "list", "--json", "llvm-openmp"],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        payload = json.loads(result.stdout) if result.stdout.strip() else []
    except json.JSONDecodeError as error:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        raise SystemExit("Conda returned invalid JSON for llvm-openmp") from error

    if result.returncode != 0:
        message = payload.get("message", "") if isinstance(payload, dict) else ""
        if "No packages match" in message:
            return []
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        raise SystemExit("Unable to inspect llvm-openmp in the active Conda environment")
    if not isinstance(payload, list):
        raise SystemExit("Unexpected Conda package metadata for llvm-openmp")
    return payload


def ensure_openmp(verbose: bool) -> None:
    '''Install llvm-openmp with Conda when it is absent from the active environment.'''
    conda_executable = shutil.which("conda")
    if conda_executable is None:
        raise SystemExit(
            "Conda is required to install llvm-openmp; activate Conda and rerun build.py"
        )
    active_prefix = os.environ.get("CONDA_PREFIX")
    if not active_prefix:
        raise SystemExit(
            "No active Conda environment detected; run conda activate seawave first"
        )
    if Path(active_prefix).resolve() != Path(sys.prefix).resolve():
        raise SystemExit(
            "The active Conda environment and Python interpreter do not match: "
            f"{active_prefix} != {sys.prefix}"
        )

    packages = conda_openmp_packages(conda_executable)
    if not packages:
        print("  llvm-openmp was not found; installing it from conda-forge")
        run(
            [
                conda_executable,
                "install",
                "--prefix",
                sys.prefix,
                "-c",
                "conda-forge",
                "llvm-openmp",
                "-y",
            ],
            verbose=verbose,
        )
        packages = conda_openmp_packages(conda_executable)
    if not packages:
        raise SystemExit("llvm-openmp is unavailable after Conda installation")
    print(f"  llvm-openmp {packages[0].get('version', 'unknown')}")


def check_prerequisites(verbose: bool) -> None:
    '''Check project files, Conda OpenMP, uv, and the active Python environment.'''
    print("[1/3] Checking prerequisites and runtime support")
    for filename in ("pyproject.toml", "README.md"):
        if not (PROJECT_ROOT / filename).is_file():
            raise SystemExit(f"Required project file is missing: {filename}")

    print(f"  Python {sys.version.split()[0]}")
    print(f"  Environment: {sys.prefix}")
    ensure_openmp(verbose)

    try:
        result = subprocess.run(
            ["uv", "--version"], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise FileNotFoundError
    except FileNotFoundError:
        print("  uv was not found; installing it in the active environment")
        run([sys.executable, "-m", "pip", "install", "uv"], verbose=verbose)
        result = subprocess.run(
            ["uv", "--version"], capture_output=True, text=True, check=False
        )
    if result.returncode != 0:
        raise SystemExit("uv is unavailable after installation")
    print(f"  {result.stdout.strip()}")


def clean_build_artifacts(verbose: bool) -> None:
    '''Remove only generated SeaWave build, metadata, and cache artifacts.'''
    print("[0/3] Cleaning generated artifacts")
    targets = [
        BUILD_DIR,
        DIST_DIR,
        PROJECT_ROOT / "seawave.egg-info",
        PROJECT_ROOT / ".pytest_cache",
    ]
    targets.extend((PROJECT_ROOT / "seawave").rglob("__pycache__"))
    for target in targets:
        if target.exists():
            shutil.rmtree(target)
            if verbose:
                print(f"  Removed {target}")
    print("  Clean completed")


def build_python(verbose: bool) -> None:
    '''Build source and wheel distributions and require a wheel artifact.'''
    print("[2/3] Building SeaWave")
    run(["uv", "build"], cwd=PROJECT_ROOT, verbose=verbose)
    wheels = list(DIST_DIR.glob("*.whl"))
    if not wheels:
        raise SystemExit("Build failed: no wheel was generated")
    for artifact in sorted(DIST_DIR.iterdir()):
        print(f"  {artifact.name}")


def install_package(verbose: bool) -> None:
    '''Install the wheel with Python dependencies and verify the full runtime.'''
    print("[3/3] Installing the newest wheel and Python dependencies")
    wheels = sorted(
        DIST_DIR.glob("*.whl"), key=lambda path: path.stat().st_mtime, reverse=True
    )
    if not wheels:
        raise SystemExit("Install failed: no wheel is available")
    run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheels[0])],
        verbose=verbose,
    )
    import_check = (
        "import jax, jaxlib, lightgbm, numpy, seawave; "
        "print(seawave.__version__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", import_check],
        cwd=PROJECT_ROOT.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("Installed package or runtime dependency import failed")
    installed_version = result.stdout.strip()
    if installed_version != read_version():
        raise SystemExit(
            f"Installed version {installed_version} does not match {read_version()}"
        )
    print("  Runtime imports passed: JAX, JAXLIB, LightGBM, NumPy, SeaWave")
    print(f"  Installed SeaWave {installed_version}")


def main() -> None:
    '''Parse build options and execute the requested build pipeline.'''
    version = read_version()
    parser = argparse.ArgumentParser(
        description=f"SeaWave {version}: build and install with dependency closure"
    )
    parser.add_argument("--install", "-i", action="store_true", help="install the wheel")
    parser.add_argument("--clean", "-c", action="store_true", help="clean before building")
    parser.add_argument("--verbose", "-v", action="store_true", help="show full output")
    args = parser.parse_args()

    print(f"\nSeaWave {version} build")
    started = time.monotonic()
    if args.clean:
        clean_build_artifacts(args.verbose)
    check_prerequisites(args.verbose)
    build_python(args.verbose)
    if args.install:
        install_package(args.verbose)
    else:
        print("[3/3] Installation skipped; use --install to install the wheel")
    print(f"Build completed in {time.monotonic() - started:.1f} seconds")


if __name__ == "__main__":
    main()
