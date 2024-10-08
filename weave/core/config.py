# weave/weave/core/config.py
import yaml
from typing import Any, Dict
import argparse


class Config:
    def __init__(self, config_path: str = None):
        self.config = {}
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def save(self, config_path: str) -> None:
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)

    @classmethod
    def from_cli(cls):
        parser = argparse.ArgumentParser(description="Weave Synthetic Data Generation Framework")
        parser.add_argument('--config', type=str, help='Path to the configuration file')
        args, _ = parser.parse_known_args()
        return cls(args.config)
