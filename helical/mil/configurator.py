import logging
import sys


class Configurator():

    def __init__(self) -> None:
        """ Creates a series of basic configs that are inherited by all other objects of this package. """
        self.log = self.get_logger()
        return
    
    def get_logger(self):
        """ Gets logger. """
        logformat = "%(asctime)s|%(levelname)s|%(message)s" # class and func name are inside message in decorator
        datefmt = "%d/%m %H:%M:%S"

        logging.basicConfig(filename="code.log", level=logging.INFO, filemode="w",
                            format=logformat, datefmt=datefmt)

        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(logging.Formatter(fmt=logformat, datefmt=datefmt))

        logger = logging.getLogger("helical")
        logger.addHandler(stream_handler)


        return logger
    
