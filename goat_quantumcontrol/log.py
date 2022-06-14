import logging
import os
import time


class LogClass:

    def __init__(self):
        self.logger = None
        self.main_path = os.getcwd()
        self.path_for_logfiles = os.path.join(self.main_path, 'Logfiles')


    def init_logger(self):
        now = time.localtime()
        current_time = time.strftime('%H_%M_%S', now)
        logname = 'log_' + current_time + '.log'
        self.logger = logging.getLogger(__name__)
        if logname not in str(self.logger.handlers):
            self.logger.handlers.clear()
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            if not os.path.exists(self.path_for_logfiles):
                os.mkdir(self.path_for_logfiles)

        # for writing to logfile
        loghandler = logging.FileHandler(os.path.join(self.path_for_logfiles, logname))
        loghandler.setFormatter(logging.Formatter('[Tracker %(levelname)s] - %(message)s'))
        self.logger.addHandler(loghandler)
        # for printing the log to the console
        consolehandler = logging.StreamHandler()
        consolehandler.setFormatter(logging.Formatter('[Tracker %(levelname)s] - %(message)s'))
        self.logger.addHandler(consolehandler)
        self.logger.propagate = False  # prevent double printing
