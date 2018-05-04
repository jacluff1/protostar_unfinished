import numpy as np

def dictionary_update(args,kwargs):
    for key in args:
        if key in kwargs: args[key] = kwargs[key]
