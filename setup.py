"""
Setup script for QFTT Bootstrap Loop package.

© 2024 - MIT License
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f 
                       if line.strip() and not line.startswith("#")]

setup(
    name="qftt-bootstrap-loop",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@institution.edu",
    description="Quantum Field Theory of Time (QFTT) Bootstrap Loop Simulation Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/qftt-bootstrap-loop",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/qftt-bootstrap-loop/issues",
        "Documentation": "https://github.com/yourusername/qftt-bootstrap-loop/docs",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=6.2.0", "pytest-cov>=2.12.0"],
        "gpu": ["cupy>=9.0.0"],
        "docs": ["sphinx>=4.0.0", "sphinx-rtd-theme>=0.5.0"],
    },
    entry_points={
        "console_scripts": [
            "qftt-run=scripts.run_simulation:main",
            "qftt-sweep=scripts.batch_parameter_sweep:main",
            "qftt-figures=scripts.generate_figures:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)