"""Setup script for Neural Trajectory Prediction project"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="neural-trajectory-prediction",
    version="0.1.0",
    author="Jad Moawad",  
    author_email="moawad.jad@gmail.com",  
    description="Neural network architectures for autonomous vehicle trajectory planning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jad-moawad/neural-trajectory-prediction",  
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
            "pre-commit>=2.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "imageio>=2.9.0",
            "imageio-ffmpeg>=0.4.5",
            "opencv-python>=4.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-trajectory=src.training.train:main",
            "evaluate-models=scripts.evaluate_models:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.md"],
    },
)