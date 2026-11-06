"""Data Transfer Objects (DTO) for worker training pipeline context and validation."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class UserInput(BaseModel):
    """Input payload providing the user training configuration file path.

    Attributes:
        user_config_train: Absolute path to the user configuration YAML.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_config_train: str = Field(
        ..., description="Path to the user training configuration YAML file"
    )


class DatasetConfig(BaseModel):
    """Configuration mapping for dataset split definitions.

    Attributes:
        data: Arbitrary dictionary configuration for the split.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: Dict[str, Any] = Field(
        default_factory=dict, description="Dataset split parameters"
    )


class Dataset(BaseModel):
    """Container holding training, validation, and test dataset split configs.

    Attributes:
        train: Training split configuration.
        val: Validation split configuration.
        test: Test split configuration.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    train: DatasetConfig
    val: DatasetConfig
    test: DatasetConfig


class DatasetUserInput(BaseModel):
    """Input container for dataset verification step.

    Attributes:
        user_config: Raw or parsed dictionary representing the user configuration.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    user_config: Any = Field(
        ..., description="Parsed user configuration dictionary or object"
    )


class PublicResultsInput(BaseModel):
    """Input container for publishing trained model results.

    Attributes:
        results_trained_model: Evaluated accuracy, fitness metric, or summary dict.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    results_trained_model: Optional[Any] = Field(
        default=None, description="Trained model fitness or accuracy score"
    )


class TrainingExecutionResult(BaseModel):
    """Execution state container representing pipeline output results.

    Attributes:
        accuracy: Top accuracy or primary fitness metric.
        status: Execution status string (e.g., 'completed', 'failed').
        timestamp: Epoch timestamp of completion.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    accuracy: float = Field(default=0.0, description="Evaluated accuracy score")
    status: str = Field(default="completed", description="Final execution status")
    timestamp: float = Field(
        default=0.0, description="Epoch timestamp of run completion"
    )
