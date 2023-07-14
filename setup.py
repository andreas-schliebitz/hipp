#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="hipp",
    version="0.1",
    extras_require={"dev": ["black"]},
    packages=find_packages(),
)
