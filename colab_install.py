"""
Google Colab one-click installer for tiles-to-mesh.

Run this script in a Colab cell to install the package:

    !python colab_install.py

Or paste the install_in_colab() function contents directly into a cell.
"""

import subprocess
import sys
import os


def install_in_colab():
    """Install tiles-to-mesh and all dependencies in Google Colab."""

    print("📦 Installing tiles-to-mesh dependencies...")
    deps = [
        "numpy>=1.24",
        "ipywidgets>=8.0",
        "pythreejs>=2.4",
        "tqdm>=4.60",
        "requests>=2.28",
        "Pillow>=9.0",
    ]
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q"] + deps,
        stdout=subprocess.DEVNULL,
    )

    # Add the python source directory to sys.path
    source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python")
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    print("✅ tiles-to-mesh installed (pure-Python mode)")
    print("   Rust acceleration: not available (optional)")
    print()
    print("   import tiles_to_mesh as ttm")


if __name__ == "__main__":
    install_in_colab()
