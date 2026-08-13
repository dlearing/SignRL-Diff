"""SignRL-Diff: RL-Guided Stable Diffusion for Sign Language Video Generation."""

from setuptools import setup, find_packages
from pathlib import Path


def read_requirements() -> list:
    """Read dependencies from requirements.txt."""
    req_path = Path(__file__).parent / "requirements.txt"
    if req_path.exists():
        return [
            line.strip()
            for line in req_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return []


def read_long_description() -> str:
    """Read the README as long description."""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return "SignRL-Diff: RL-Guided Stable Diffusion for Sign Language Video Generation"


setup(
    name="signrl-diff",
    version="0.1.0",
    description="RL-guided video diffusion model for sign language generation",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="SignRL-Diff Contributors",
    license="MIT",
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "signrl-train-phase1=signrl_diff.scripts.train_phase1:main",
            "signrl-train-phase2=signrl_diff.scripts.train_phase2:main",
            "signrl-train-phase3=signrl_diff.scripts.train_phase3:main",
            "signrl-infer=signrl_diff.scripts.inference:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
