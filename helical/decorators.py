import time
from loggers import get_logger

def timer(inner_func):
    """ Decorator to compute duration time of a function. """
    def wrapped_func(*args, **kwargs):
        start = time.time()
        inner_func(*args, **kwargs)
        end = time.time()
        duration = end-start
        print(f"Executed in {duration:.3f} secs.")
    return wrapped_func

def log_start_finish(class_name:str, func_name:str, msg:str):
    def inner_decorator(orig_func):
        def wrapped_func(*args, **kwargs):
            log = get_logger()
            try:
                log.info(f"{class_name}.{func_name}: ⏳{msg}") #, extra={'className': self.__class__.__name__
                returned = orig_func(*args, **kwargs)
                log.info(f"{class_name}.{func_name}: ✅{msg}") #, extra={'className': self.__class__.__name__
            except:
                log.error(f"{class_name}.{func_name}: ❌{msg}") #, extra={'className': self.__class__.__name__
                returned = None
            return returned

        return wrapped_func
    return inner_decorator






@log_start_finish(class_name='Boh', func_name='_normalize_file', msg='Normalizing')
def my_func(sleep_time:int): 
    time.sleep(sleep_time)
    print('done')
    return




if __name__ == '__main__':
    my_func(1)

