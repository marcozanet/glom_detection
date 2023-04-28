from loggers import get_logger


class Configurator():

    def __init__(self) -> None:
        """ Creates a series of basic configs that are inherited by all other objects of this package. """
        self.log = get_logger()
        return


