"""This file is used to override the default config values in config.py"""
import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if "=" not in arg:
        assert not arg.startswith("--")
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        assert arg.startswith("--")
        key, val = arg.split("=")
        key = key[2:]
        if key in globals():
            try:
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val
            assert type(attempt) == type(globals()[key])
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
