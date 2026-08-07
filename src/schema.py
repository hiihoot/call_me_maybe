"""Schema parsing and validation for function definitions and prompts."""
import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, ValidationError


class FunctionCallSchema(BaseModel):
    """Pydantic model for a single function definition."""

    model_config = ConfigDict(strict=True, extra="ignore")
    name: str
    description: str
    parameters: Dict[str, Any]
    returns: Dict[str, Any]


def _read_json(path: str) -> Any:
    """Read and parse a JSON file with thorough validation.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON data.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be read.
        ValueError: If the file is empty or contains invalid JSON.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    try:
        raw = file_path.read_text(encoding="utf-8")
    except PermissionError:
        raise PermissionError(
            f"Permission denied: cannot read {path}. "
            f"Check file permissions or run with appropriate access."
        ) from None
    except OSError as exc:
        raise OSError(f"Cannot read file {path}: {exc}") from exc

    if not raw.strip():
        raise ValueError(f"File is empty: {path}")

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {path} at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc


def parse_definitions(path: str) -> List[FunctionCallSchema]:
    """Parse and validate function definitions from a JSON file.

    Args:
        path: Path to the functions definition JSON file.

    Returns:
        A list of validated FunctionCallSchema objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be read.
        ValueError: If the JSON is invalid, malformed, or schema validation fails.
    """
    data = _read_json(path)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array in {path}, got {type(data).__name__}"
        )
    if not data:
        raise ValueError(f"Function definitions array is empty: {path}")

    parsed: List[FunctionCallSchema] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"Function definition at index {idx} must be an object, "
                f"got {type(item).__name__}"
            )
        if "name" not in item:
            raise ValueError(
                f"Function definition at index {idx} is missing required "
                f"key 'name'"
            )
        try:
            parsed.append(FunctionCallSchema(**item))
        except ValidationError as exc:
            raise ValueError(
                f"Schema validation failed for function "
                f"'{item.get('name', '<unknown>')}' at index {idx}: {exc}"
            ) from exc
    return parsed


def parse_prompts(path: str) -> List[str]:
    """Parse test prompts from a JSON file.

    Args:
        path: Path to the prompts JSON file.

    Returns:
        A list of prompt strings.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be read.
        ValueError: If the JSON is invalid or malformed.
    """
    data = _read_json(path)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array in {path}, got {type(data).__name__}"
        )
    if not data:
        raise ValueError(f"Prompts array is empty: {path}")

    prompts: List[str] = []
    for idx, item in enumerate(data):
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict):
            if "prompt" not in item:
                raise ValueError(
                    f"Prompt object at index {idx} missing required "
                    f"key 'prompt'. Keys found: {list(item.keys())}"
                )
            if not isinstance(item["prompt"], str):
                raise ValueError(
                    f"Prompt at index {idx} must be a string, "
                    f"got {type(item['prompt']).__name__}"
                )
            prompts.append(item["prompt"])
        else:
            raise ValueError(
                f"Prompt item at index {idx} must be a string or object, "
                f"got {type(item).__name__}"
            )
    return prompts