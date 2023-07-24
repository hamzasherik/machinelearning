import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())
logger.handlers[0].setFormatter(logging.Formatter('%(levelname)s: %(funcName)s [Line: %(lineno)d]: %(message)s'))