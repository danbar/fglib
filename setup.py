from setuptools import setup, find_packages

import fglib

setup(
    name="fglib",
    version=fglib.__version__,
    packages=find_packages(),
    scripts=[],

    # Dependencies
    install_requires=["networkx>=2.0",
                      "numpy>=1.8",
                      "matplotlib>=1.3"],

    # Metadata
    author="Daniel Bartel",
    description="The factor graph library (fglib) is a Python 3 package to simulate message passing on factor graphs.",
    license="MIT License",
    keywords="factor graph message passing",
    url="https://github.com/danbar/fglib/",

    # Miscellaneous
    test_suite="nose.collector"
)
