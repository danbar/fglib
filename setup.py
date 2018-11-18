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
    python_requires='>=3.4',
    install_requires=["networkx>=2.0",
                      "numpy>=1.12",
                      "matplotlib>=2.0"],

    # Metadata
    author="Daniel Bartel",
    author_email='dan.bar@gmx.at',
    description="factor graph library",
    long_description=long_description,
    license="MIT License",
    keywords="factor graph message passing",
    url="https://github.com/danbar/fglib/",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    # Miscellaneous
    test_suite="nose.collector"
)
