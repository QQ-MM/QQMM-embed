import sys
import logging


class LevelFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, level_fmts=None):
        self._level_formatters = {}
        if level_fmts is None:
            level_fmts = {}
        for level_, fmt_ in level_fmts.items():
            self._level_formatters[level_] = logging.Formatter(fmt=fmt_, datefmt=datefmt)
        super(LevelFormatter, self).__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record):
        if record.levelno in self._level_formatters:
            return self._level_formatters[record.levelno].format(record)

        return super(LevelFormatter, self).format(record)


def get_logger(logger_name=''):
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.NOTSET)
    # formatter = LevelFormatter(datefmt='%H:%M:%S',
    #                            level_fmts={logging.CRITICAL: '%(asctime)s %(name)s %(levelname)s %(message)s',
    #                                        logging.ERROR: '%(asctime)s %(name)s %(levelname)s %(message)s',
    #                                        logging.WARNING: '%(asctime)s %(name)s %(levelname)s %(message)s'})
    # console.setFormatter(formatter)
    logger = logging.getLogger(logger_name)
    logger.addHandler(console)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    return logger


logger = get_logger('QQMM')
