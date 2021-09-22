import logging
import time
from functools import wraps
from typing import Final


def timing(f):
    """Decorator for timing functions
    Usage:
    @timing
    def function(a):
        pass
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start: Final = time.time()
        result: Final = f(*args, **kwargs)
        end: Final = time.time()
        logging.info("function:%r took: %2.5f sec" % (f.__name__, end - start))
        return result

    return wrapper
