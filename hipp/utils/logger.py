#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging


class Logger:
    def __init__(self,
                 name,
                 format: str = '%(asctime)s - %(levelname)s - %(message)s',
                 level: int = logging.INFO):
        self.name: str = name
        self.format: str = format
        self.level: int = level
        logging.basicConfig(format=format, level=level)

    def _msg(self, method_name: str, message: str) -> str:
        return f'[{self.name}::{method_name}] - {message}'

    def info(self, method_name: str, message: str) -> None:
        logging.info(self._msg(method_name, message))

    def error(self, method_name: str, message: str) -> None:
        logging.error(self._msg(method_name, message))

    def warning(self, method_name: str, message: str) -> None:
        logging.warning(self._msg(method_name, message))

    def __str__(self) -> str:
        return f'Logger(name={self.name}, format={self.format}, level={self.level})'
