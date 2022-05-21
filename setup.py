from codecs import open
from os import path

from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(here, 'requirements.txt')) as f:
    install_requires = [x for x in f.read().splitlines() if len(x)]

setup(
    name="tf-SmeLU",
    version="1.0.0",
    description="Tensorflow Smooth ReLU (SmeLU) implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/r3v1/tf-SmeLU",
    author="David",
    author_email="r3v1@pm.me",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    # Note that this is a string of words separated by whitespace, not a list.
    keywords="tensorflow activation smelu",
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    install_requires=install_requires,
    extras_require={
        "matplotlib": ["matplotlib"]
    },
    python_requires=">=3.8",
    license="MIT",
)
