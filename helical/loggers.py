import logging
import sys

def get_logger():
    """ Gets logger. """
    logformat = "%(asctime)s %(levelname)s - %(funcName)s: %(message)s"
    datefmt = "%d/%m %H:%M"

    logging.basicConfig(filename="code.log", level=logging.INFO, filemode="w",
                        format=logformat, datefmt=datefmt)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(logging.Formatter(fmt=logformat, datefmt=datefmt))

    logger = logging.getLogger("helical")
    logger.addHandler(stream_handler)


    return logger


def fun():
    logger = get_logger()
    logger.info(" fun inf")
    logger.warning("fun warn")


if __name__ == "__main__":
    fun()