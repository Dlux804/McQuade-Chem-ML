# File for random commands


# to change the working directory inside of a context manager
# it will revert to previous directory when finished with loop.
from contextlib import contextmanager
import os


@contextmanager
def cd(newdir):
    """
    Change the working directory inside of a context manager.
    It will revert to previous directory when finished with loop.
    """
    prevdir = os.getcwd()
    # print('Previous PATH:', prevdir)
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        # print('Switching back to previous PATH:', prevdir)
        os.chdir(prevdir)

# Example usage
'''
os.chdir('/home')

with cd('/tmp'):
    # ...
    raise Exception("There's no place like home.")
# Directory is now back to '/home'.
'''

