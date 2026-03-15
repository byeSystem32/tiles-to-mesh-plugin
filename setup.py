"""
Pure-Python fallback installer for tiles-to-mesh.

This allows installing the package without the Rust toolchain (e.g. in Google Colab).
The Rust extension is optional — when unavailable, the package falls back to a
pure-Python implementation that is functionally identical but slower.

Usage (Colab / no Rust):
    pip install .

Usage (with Rust, for maximum performance):
    pip install maturin && maturin develop --release
"""

from setuptools import setup, find_packages

setup(
    name="tiles-to-mesh",
    version="0.1.0",
    description="Fetch Google 3D Tiles, build meshes, clean up with Blender, and preview in Jupyter/Colab",
    python_requires=">=3.9",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    install_requires=[
        "numpy>=1.24",
        "ipywidgets>=8.0",
        "pythreejs>=2.4",
        "tqdm>=4.60",
        "requests>=2.28",
        "Pillow>=9.0",
    ],
    extras_require={
        "blender": ["bpy>=4.0"],
    },
)
