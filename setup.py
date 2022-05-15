"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

name = "mil"
version = "0.2"
release = "0.1.0"
description = "Multiple Instance Learning"
author = "Nima"
author_email = "nima.manaf8@gmail.com"
url = "nimaman.github.io"

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()


dev_requirements = ["black==20.8b1"]


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name=name,
    version=release,
    description=description,
    long_description=long_description,
    url=url,
    author=author,
    author_email=author_email,
    # For a list of valid classifiers, see
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(exclude=["doc", "tests"]),
    #  options for documentation builder
    command_options={
        "build_sphinx": {
            "project": ("setup.py", name),
            "version": ("setup.py", version),
            "release": ("setup.py", release),
            "copyright": ("setup.py", copyright),
            "source_dir": ("setup.py", "doc/source"),
            "build_dir": ("setup.py", "doc/build"),
        }
    },
    python_requires=">=3.6",
    setup_requires=["pytest-runner", "setuptools-git-version", "wheel"],
    install_requires=requirements,
    extras_require={"dev": dev_requirements},
)