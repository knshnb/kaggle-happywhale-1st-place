from typing import Optional

import yaml


class Config(dict):
    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError:
            return super().__getattr__(key)
        if isinstance(val, dict):
            return Config(val)
        return val


def load_config(path: str, default_path: Optional[str]) -> Config:
    with open(path) as f:
        cfg = Config(yaml.full_load(f))
    if default_path is not None:
        # set keys not included in `path` by default
        with open(default_path) as f:
            default_cfg = Config(yaml.full_load(f))
        for key, val in default_cfg.items():
            if key not in cfg:
                print(f"used default config {key}: {val}")
                cfg[key] = val
    return cfg
