"""
Setup script for SigClean - A library for cleaning biomedical signals.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt if it exists
requirements = [
    'numpy>=1.19.0',
    'scipy>=1.6.0',
    'matplotlib>=3.3.0',
]

# Optional requirements for development
dev_requirements = [
    'pytest>=6.0',
    'pytest-cov>=2.10',
    'black>=21.0',
    'flake8>=3.8',
    'sphinx>=3.5',
    'sphinx-rtd-theme>=0.5',
]

setup(
    name="sigclean",
    version="1.0.0",
    author="Diptiman Mohanta",
    author_email="diptimanmohanta7@gmail.com",
    description="A comprehensive Python library for cleaning and preprocessing biomedical signals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/diptiman-mohanta/sigclean",
    project_urls={
        "Bug Tracker": "https://github.com/diptiman-mohanta/sigclean/issues",
        "Documentation": "https://sigclean.readthedocs.io/",
        "Source Code": "https://github.com/diptiman-mohanta/sigclean",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Signal Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "test": ["pytest>=6.0", "pytest-cov>=2.10"],
        "docs": ["sphinx>=3.5", "sphinx-rtd-theme>=0.5"],
    },
    keywords=[
        "biomedical", "signal processing", "ECG", "EMG", "EEG", 
        "filtering", "preprocessing", "noise reduction", "artifact removal"
    ],
    include_package_data=True,
    zip_safe=False,
    test_suite="tests",
    tests_require=["pytest>=6.0"],
    entry_points={
        "console_scripts": [
            # Add console scripts here if needed
            # "sigclean-cli=sigclean.cli:main",
        ],
    },
)