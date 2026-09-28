"""Synthetic Data Generator: realistic datasets from a schema, powered by Claude structured outputs."""

__version__ = "1.0.0"

from .generator import GenerationError, Stats, TableGenerator, generate, generate_dataset  # noqa: E402
from .schema_builder import DatasetSchema, SchemaError, TableSpec, load_schema, schema_from_dict  # noqa: E402
from .validator import format_report, quality_report, validate_record  # noqa: E402

__all__ = [
    "generate", "generate_dataset", "TableGenerator", "GenerationError", "Stats",
    "DatasetSchema", "TableSpec", "SchemaError", "load_schema", "schema_from_dict",
    "quality_report", "format_report", "validate_record", "__version__",
]
