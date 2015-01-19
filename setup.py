from setuptools import setup, find_packages

import fglib

setup(
    name="fglib",
    version=fglib.__version__,
    url="https://github.com/danbar/fglib/",
    license="MIT License",
    author="Daniel Bartel",
    install_requires=["numpy>=1.9"],
    description="factor graph library",
    packages=find_packages(),
    test_suite="nose.collector"
)
