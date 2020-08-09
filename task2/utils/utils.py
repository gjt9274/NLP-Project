import json


def read_json(jsonfile):
    with open(jsonfile, 'r') as f:
        cfg = json.load(f)
    return cfg


class ConfigParser(object):
    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, k):
        try:
            v = self._cfg[k]
        except KeyError:
            return super().__getattribute__(k)
        if isinstance(v, dict):
            return ConfigParser(v)
        return v
