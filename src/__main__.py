"""Main entry point for the function calling pipeline."""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from rich import print as rprint

from src.generator import Generator
from src.schema import parse_definitions, parse_prompts


def main() -> int:
    """Execute the function calling pipeline.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = argparse.ArgumentParser(
        description="Run function calling constrained decoding pipeline."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to functions definition JSON file.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to test prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to output JSON file.",
    )
    args = parser.parse_args()

    # --- Parse inputs ---
    try:
        definitions = parse_definitions(args.functions_definition)
    except FileNotFoundError as exc:
        rprint(f"[bold red]File not found:[/bold red] {exc}")
        return 1
    except PermissionError as exc:
        rprint(f"[bold red]Permission denied:[/bold red] {exc}")
        return 1
    except ValueError as exc:
        rprint(f"[bold red]Invalid function definitions:[/bold red] {exc}")
        return 1

    try:
        prompts = parse_prompts(args.input)
    except FileNotFoundError as exc:
        rprint(f"[bold red]File not found:[/bold red] {exc}")
        return 1
    except PermissionError as exc:
        rprint(f"[bold red]Permission denied:[/bold red] {exc}")
        return 1
    except ValueError as exc:
        rprint(f"[bold red]Invalid prompts file:[/bold red] {exc}")
        return 1

    # --- Initialize generator ---
    try:
        gen = Generator(definitions)
    except FileNotFoundError as exc:
        rprint(f"[bold red]Missing resource:[/bold red] {exc}")
        return 1
    except ValueError as exc:
        rprint(f"[bold red]Invalid resource:[/bold red] {exc}")
        return 1
    except RuntimeError as exc:
        rprint(f"[bold red]Generator init failed:[/bold red] {exc}")
        return 1

    results: List[Dict[str, Any]] = []
    rprint("[bold green]Starting Function Calling Generation...[/bold green]")

    for idx, user_prompt in enumerate(prompts):
        rprint(
            f"[cyan]Processing {idx + 1}/{len(prompts)}:[/cyan] "
            f"'{user_prompt}'"
        )
        try:
            result = gen.run_prompt(user_prompt)
        except Exception as exc:
            rprint(
                f"[yellow]Warning:[/yellow] Failed to process prompt "
                f"{idx + 1}: {exc}"
            )
            result = {
                "prompt": user_prompt,
                "name": "",
                "parameters": {},
            }
        results.append(result)

    # --- Write output ---
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except PermissionError:
        rprint(
            f"[bold red]Permission denied:[/bold red] "
            f"Cannot write to {args.output}"
        )
        return 1
    except OSError as exc:
        rprint(f"[bold red]Write error:[/bold red] {exc}")
        return 1

    rprint(
        f"[bold green]Successfully saved results to {args.output}[/bold green]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
