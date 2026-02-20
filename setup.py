

import setuptools
import os


# Get this file path
current_path = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(current_path, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open(os.path.join(current_path, "requirements.txt"), "r", encoding="utf-8") as fh:
    install_requires = fh.read().splitlines()

setuptools.setup(
    name="lpspline",
    version="0.0.3",
    author="clarkmaio",
    author_email="maioliandrea0@gmail.com",
    description="lpspline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/clarkmaio/lpspline",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires='>=3.10',
    install_requires=install_requires,
    keywords="",
)


