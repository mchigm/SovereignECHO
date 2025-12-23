#./src/main.py

'''

main.py

Author: MCHIGM

Date: 2025/12/13

'''

################################################## Setup ##################################################



# =============
# Get ROOT
# =============


#import necessary modules
from pathlib import Path
import sys

#detect/get ROOT path
root_PATH = Path(__file__).resolve().parents[1]


# ===============
# Import library
# ===============


#insert library path to local python system path
sys.path.insert(0, str(root_PATH / "lib"))

#import lib module
from mod import *


# ===============
# Import modules
# ===============


#import built-in modules
import os, sys, time                                                                       # Basic functions
import multiprocessing as mp                                                               # Multiprocessing
import threading                                                                           # Multithreading

#import environment modules
try:                                                                                       # Try to import package
    ...
except ImportError:                                                                        # If package have problem while importing
    ...
finally:                                                                                   # Perform action to clean up
    os.system('cls')                                                               # Clear console

################################################## Methods ##################################################



# =================
# Common functions
# =================


def example_functin(x) -> any:
    raise NotImplementedError


# =============
# Decorators
# =============


def example_decorator(func:function) -> function:
    def wrapper(*args:tuple, **kwargs:dict) -> any:
        result = func(*args, **kwargs)
        
        return result
    
    return wrapper


# =============
# Classes
# =============


class Example_Class:

    def __init__(self) -> None:
        pass

    pass


# =============
# Main method
# =============


...



# ========================
# Supplementary functions
# ========================


#advanced garbage collection
def start_timed_garbage_collection(interval:float=60.0) -> None:
    """ Starts a time garbage collection loop that runs every period of time (in minutes).

    Args:
        interval (float): The time interval (in minutes) between each garbage collection cycle. Default is 60.0 minutes.  
    
    
    After calling this function, the function will start a parallel timer.  Each interval, garbage collection will be triggered.  

    Example:
        start_timed_garbage_collection(interval=30.0)  # Starts garbage collection every 30 minutes
    """
    def gc_loop() -> sys.NoReturn:
        while True:
            time.sleep(interval * 60)                                            # Convert minutes to seconds
            gc.collect()                                                         # Collect garbage
    
    gc_thread = threading.Thread(target=gc_loop, daemon=True)                    # Create a new thread
    gc_thread.start()                                                            # Start garbage collect routine

#simple memory management
...

################################################### Main ###################################################



# =============
# Setup
# =============


...


# =============
# Entry point
# =============


...


# =============
# Stream
# =============


if __name__ == "__main__":                                                                            # Main
    pass


# =============
# Loop
# =============


...


# =============
# Cleanup
# =============


...