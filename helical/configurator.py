import logging
import sys
from typing import Type


class Configurator():

    def __init__(self) -> None:
        """ Creates a series of basic configs that are inherited 
            by all other objects of this package. """
        self.log = self.get_logger()
        return
    

    def get_logger(self)->Type:
        """ Gets logger. """
        logformat = "%(asctime)s|%(levelname)s|%(message)s"
        datefmt = "%d/%m %H:%M:%S"
        logging.basicConfig(filename="code.log", level=logging.INFO, filemode="w",
                            format=logformat, datefmt=datefmt)

        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(logging.Formatter(fmt=logformat, datefmt=datefmt))
        stream_handler.terminator = "" #TODO DELETE
        logger = logging.getLogger("helical")
        logger.addHandler(stream_handler)
        return logger
    
        
    def format_msg(self, msg:str, func_n:str, type:str='info')->str:
        """ Formats the message to be logged with class name and method name. """
        _get_msg_base = lambda func_n: f"{self.__class__.__name__}.{func_n}: "
        if type=='info':
            return self.log.info(_get_msg_base(func_n=func_n) + msg)
        elif type=='warning':
            return self.log.warning(_get_msg_base(func_n=func_n) + msg)
        elif type=='error':
            return self.log.error(_get_msg_base(func_n=func_n) + msg)
        else:
            raise NotImplementedError()
    

    def assert_log(self, msg:str, func_n:str)->str:
        """ Logs an assertion error with formatted message and raises an Assertion Error. """
        _get_msg_base = lambda func_n: f"{self.__class__.__name__}.{func_n}: "
        self.log.error(_get_msg_base(func_n=func_n) + msg)
        raise AssertionError(msg)
    
