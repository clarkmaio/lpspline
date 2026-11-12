

import os
import re

import setuptools


# Get this file path
current_path = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_path, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Single source of truth for the version: lpspline/__init__.py
with open(os.path.join(current_path, "lpspline", "__init__.py"), "r", encoding="utf-8") as fh:
    version = re.search(r'^__version__ = "([^"]+)"', fh.read(), re.M).group(1)

# Read requirements from requirements.txt
with open(os.path.join(current_path, "requirements.txt"), "r", encoding="utf-8") as fh:
    install_requires = fh.read().splitlines()

setuptools.setup(
    name="lpspline",
    version=version,
    author="clarkmaio",
    author_email="maioliandrea0@gmail.com",
    description="lpspline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://lpspline.readthedocs.io/en/latest/",
    project_urls={"Source": "https://github.com/clarkmaio/lpspline"},
    packages=setuptools.find_packages(include=["lpspline", "lpspline.*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires='>=3.10',
    install_requires=install_requires,
    keywords="",
)


