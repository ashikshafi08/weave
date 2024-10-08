# weave/weave/core/config.py
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import json

class LLMProviderConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)

class DataGeneratorConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)

class TaskCreatorConfig(BaseModel):
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)

class WeaveConfig(BaseModel):
    data_generator: DataGeneratorConfig
    task_creator: TaskCreatorConfig
    llm_provider: LLMProviderConfig
    pipeline_stages: List[str] = Field(default_factory=list)
    num_samples: int = 100

    @classmethod
    def from_json(cls, file_path: str):
        with open(file_path, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)

    def to_json(self, file_path: str):
        with open(file_path, 'w') as f:
            json.dump(self.dict(), f, indent=2)

class Config:
    def __init__(self, config_path: str = None):
        self.config = WeaveConfig.from_json(config_path) if config_path else WeaveConfig()

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.dict().get(key, default)

    def set(self, key: str, value: Any) -> None:
        setattr(self.config, key, value)

    def save(self, config_path: str) -> None:
        self.config.to_json(config_path)