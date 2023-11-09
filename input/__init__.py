from yaml import safe_load
import os

filepath = os.path.join(os.getcwd(), 'input/paramfile.yml')

with open(filepath, 'r') as f:
    params = safe_load(f)

def update_parameters(params, keys, value):
    if len(keys) == 1:
        params[keys[0]] = value
    else:
        update_parameters(params[keys[0]], keys[1:], value)