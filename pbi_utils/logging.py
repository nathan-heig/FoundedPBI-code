import logging
import sys
from colorama import Fore, Back, Style

TRACE = 5
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR

class Logging(object):
    """
    To log a message, at the start of your main program, add the following lines to configure the level of logging that will be displayed:
    ```
    from utils.logging import Logging, INFO
    Logging.set_logging_level(INFO)             # Change INFO with the minimum level that you want to log
    ```

    and then, wherever you want to log, simply do:
    ```
    logger = Logging()

    logger.trace("This is a trace message")
    logger.debug("This is a debug message")
    logger.info("This is a info message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    """

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Logging, cls).__new__(cls)
        return cls.instance

    def __init__(self, disable_external_logging=True) -> None:
        self.debug_logger = None
        self.info_logger = None
        self.warning_logger = None
        self.error_logger = None
        self.trace_logger = None
        self.disable_external_logging = disable_external_logging

        if self.disable_external_logging: self.disable()

    @staticmethod
    def set_logging_level(level = INFO): 
        """Set the minimum logging level that will be displayed. All messages below this level will be hidden.
        
        You only need to do this once during the life of the application, usually at the beginning.
        
        Use like:
        ```
        from utils.logging import Logging, INFO
        Logging.set_logging_level(INFO)
        """

        logging.basicConfig(level=level, stream=sys.stdout, force=True)
    
    def __create_new_logger(self, level: int, format: str):
        # Logger setup (https://middleware.io/blog/python-logging-format/)
        logger = logging.getLogger(str(level))
        console_handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter(fmt=format)
        logger.propagate = False  # To avoid repeated outputs
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger
    
    def trace(self, msg) -> None:
        self.enable()
        if not self.trace_logger:
            self.trace_logger = self.__create_new_logger(TRACE, format=f"{Fore.MAGENTA}[%(levelname)s]{Fore.RESET} %(message)s")
        self.trace_logger.log(TRACE, msg)
        if self.disable_external_logging: self.disable()

    def debug(self, msg) -> None:
        self.enable()
        if not self.debug_logger:
            self.debug_logger = self.__create_new_logger(DEBUG, format=f"{Fore.CYAN}[%(levelname)s]{Fore.RESET} %(message)s")
        self.debug_logger.debug(msg)
        if self.disable_external_logging: self.disable()

    def info(self, msg) -> None:
        self.enable()
        if not self.info_logger:
            self.info_logger = self.__create_new_logger(INFO, format=f"{Fore.GREEN}[%(levelname)s]{Fore.RESET} %(message)s")
        self.info_logger.info(msg)
        if self.disable_external_logging: self.disable()

    def warning(self, msg) -> None:
        self.enable()
        if not self.warning_logger:
            self.warning_logger = self.__create_new_logger(WARNING, format=f"{Fore.YELLOW}[%(levelname)s]{Fore.RESET} %(message)s")
        self.warning_logger.warning(msg)
        if self.disable_external_logging: self.disable()

    def error(self, msg) -> None:
        self.enable()
        if not self.error_logger:
            self.error_logger = self.__create_new_logger(ERROR, format=f"{Fore.RED}[%(levelname)s]{Fore.RESET} %(message)s")
        self.error_logger.error(msg)
        if self.disable_external_logging: self.disable()

    def disable(self) -> None:
        """Disable all logging messages."""
        logging.disable(logging.CRITICAL)
    
    def enable(self) -> None:
        """Enable logging messages."""
        logging.disable(logging.NOTSET)