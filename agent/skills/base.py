"""Shared interface for every agent skill (LangGraph nodes in Week 3 call skills this way)."""
from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel


class BaseAgentSkill(ABC):
    name: ClassVar[str]
    description: ClassVar[str] = ""
    input_model: ClassVar[type[BaseModel]]
    output_model: ClassVar[type[BaseModel]]

    def run(self, payload: dict) -> dict:
        """Validate input -> execute -> validate output -> plain dict (JSON-safe)."""
        data = self.input_model.model_validate(payload)
        result = self.execute(data)
        if isinstance(result, BaseModel):
            result = result.model_dump()
        return self.output_model.model_validate(result).model_dump(mode="json")

    @abstractmethod
    def execute(self, data: BaseModel) -> BaseModel | dict:
        ...
