"""Logger"""


# Standard dist imports
import datetime
import logging
import os

class Logger(object):
    """Logger object for dataset generation and model training/testing"""

    def __init__(self, log_filename, level, log2file=False):
        """ Initializes logger
        Args:
            log_filename: The full path directory to log file
        """
        self.log_filename = log_filename
        dir_name = os.path.dirname(log_filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        if not log2file:
            logging.basicConfig(level=level,
                                format='',
                                datefmt='%m-%d %H:%M:S',
                                filename=log_filename,
                                filemode='w')

            # define a Handler which writes LVL messages or higher to the sys.stderr
            console = logging.StreamHandler()
            console.setLevel(level)
            # set a format which is simpler for console use
            formatter = logging.Formatter('')
            # tell the handler to use this format
            console.setFormatter(formatter)
            # add the handler to the root logger
            logging.getLogger('').addHandler(console)
        else:
            logging.basicConfig(filename=log_filename, level=logging.DEBUG,
                                format='', filemode='w')

        logging.info('Logger initialized @ {}'.format(datetime.datetime.now()))

    @staticmethod
    def section_break(title):
        logging.debug("="*30 + "   {}   ".format(title) + "="*30)

if __name__ == '__main__':
    """Example for usage"""
    log_filename = 'data/test.log'
    Logger(log_filename, logging.DEBUG, False)

    Logger.section_break(title='Arguments')
    logger = logging.getLogger('test')
    logger.info('test')