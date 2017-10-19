from setuptools import setup, find_packages

import fglib

long_description = ("The factor graph library (fglib) is a Python package "
                    "to simulate message passing on factor graphs.")

setup(
    name="fglib",
    version=fglib.__version__,
    packages=find_packages(),
    scripts=[],

    # Dependencies
    install_requires=["networkx>=2.0",
                      "numpy>=1.12",
                      "matplotlib>=2.0"],

    # Metadata
    author="Daniel Bartel",
    description="factor graph library",
    long_description=long_description,
    license="MIT License",
    keywords="factor graph message passing",
    url="https://github.com/danbar/fglib/",

    # Miscellaneous
    test_suite="nose.collector"
)
