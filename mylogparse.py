#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'yp'


import logging
from functools import wraps
from datetime import timedelta, datetime


class LogParse(object):
    def __init__(self):
        self.delay = None
        self.logger = None
        self.handler = None
        self.filename = None
        self.path = None
        self.message = None

    def set_profile(self, path, filename, delay=0):
        self.path = path
        self.delay = delay

        today = datetime.today()
        mission_time = today + timedelta(days=-self.delay)
        mission_day = mission_time.strftime('%Y%m%d')

        self.filename = "{0}/{1}_{2}.logs".format(self.path, filename, mission_day)

        logger = logging.getLogger('logger')
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format='%(asctime)s:%(levelname)s:(info) %(message)s', level=logging.DEBUG)

        handler = logging.FileHandler(self.filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:(file) %(message)s'))

        logger.addHandler(handler)
        self.logger = logger
        self.handler = handler

    def exception(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as error:
                # logs the exception
                err = "There was an exception in  "
                err += func.__name__
                self.logger.exception(error)

        return wrapper

    def info(self, message):
        self.message = message
        self.logger.info(self.message)
