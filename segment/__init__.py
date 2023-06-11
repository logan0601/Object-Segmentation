__modules__ = {}


def register(name):
    def decorator(cls):
        __modules__[name] = cls
        return cls

    return decorator


def find(name):
    return __modules__[name]


###  grammar sugar for logging utilities  ###
import logging

logger = logging.getLogger("object_segmentation")


def debug(*args, **kwargs):
    logger.debug(*args, **kwargs)


def info(*args, **kwargs):
    logger.info(*args, **kwargs)


def warn(*args, **kwargs):
    logger.warn(*args, **kwargs)


from . import data, models, utils
