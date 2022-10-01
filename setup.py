#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# https://stackoverflow.com/questions/28509965/setuptools-development-requirements
setup(
    name="Hyperspectral Image Preprocessing",
    version="0.1",
    extras_require={"dev": ["black"]},
    packages=find_packages(),
)
