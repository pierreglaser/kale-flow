#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

description = (
    "Implementation of wasserstein gradient flow of measure-embedding "
    "functionals"
)

dist = setup(
    name="kernel_wasserstein_flows",
    version="0.0.1dev0",
    description=description,
    author="Pierre Glaser",
    author_email="pierreglaser@msn.com",
    license="BSD 3-Clause License",
    packages=["kernel_wasserstein_flows"],
    install_requires=["numpy", "torch"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    python_requires=">=3.6",
)
