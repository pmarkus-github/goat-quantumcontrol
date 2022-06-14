import logging
import os
import time

global logger
main_path = os.getcwd()
path_for_logfiles = os.path.join(main_path, 'Logfiles')


def init_logger():
    now = time.localtime()
    current_time = time.strftime('%H_%M_%S', now)
    logname = 'log_' + current_time + '.log'
    logger = logging.getLogger(__name__)
    if logname not in str(logger.handlers):
        logger.handlers.clear()
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        if not os.path.exists(path_for_logfiles):
            os.mkdir(path_for_logfiles)

    # for writing to logfile
    loghandler = logging.FileHandler(os.path.join(path_for_logfiles, logname))
    loghandler.setFormatter(logging.Formatter('[Tracker %(levelname)s] - %(message)s'))
    logger.addHandler(loghandler)
    # for printing the log to the console
    consolehandler = logging.StreamHandler()
    consolehandler.setFormatter(logging.Formatter('[Tracker %(levelname)s] - %(message)s'))
    logger.addHandler(consolehandler)
    logger.propagate = False  # prevent double printing
