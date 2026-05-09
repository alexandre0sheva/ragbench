from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RunConfig(BaseModel):
    name: str = "ragbench_run"
    output_dir: Path = Path("results")


class DatasetConfig(BaseModel):
    documents_path: Path
    questions_path: Path
    qrels_path: Path | None = None


class SystemConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    name: str | None = None
    chunker: dict[str, Any] = Field(default_factory=dict)
    retrieval: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, Any] = Field(default_factory=dict)
    llm_features: dict[str, Any] = Field(default_factory=dict)

    @property
    def resolved_name(self) -> str:
        return self.name or self.type


class EvaluationConfig(BaseModel):
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    judge_enabled: bool = True
    judge_model: str = "gpt-5.4-nano"
    max_questions: int | None = None
    max_workers: int = 4


class ExperimentConfig(BaseModel):
    run: RunConfig
    dataset: DatasetConfig
    systems: list[SystemConfig]
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
