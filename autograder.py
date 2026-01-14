"""Autograder for extracting and validating temperature guesses from model responses."""

import re
import json
from pathlib import Path


def extract_temperature(response: str) -> float | None:
    """Extract temperature from model response using the required format.

    Looks for pattern: TEMPERATURE: <number>
    Returns the number if valid (0-1 range), None otherwise.
    """
    # Match TEMPERATURE: followed by a number (int or float)
    match = re.search(r'TEMPERATURE:\s*([0-9]*\.?[0-9]+)', response, re.IGNORECASE)
    if match:
        try:
            value = float(match.group(1))
            if 0 <= value <= 1:
                return value
        except ValueError:
            pass
    return None


def grade_single_result(result: dict) -> dict:
    """Grade a single result by extracting the temperature guess."""
    parsed = extract_temperature(result['response'])
    return {
        **result,
        'parsed_temperature': parsed,
        'parse_success': parsed is not None,
        'error': abs(parsed - result['actual_temperature']) if parsed is not None else None
    }


def grade_results(results: list[dict]) -> list[dict]:
    """Grade all results and add parsed temperature fields."""
    return [grade_single_result(r) for r in results]


def grade_results_file(input_file: str, output_file: str | None = None) -> list[dict]:
    """Grade results from a JSON file and optionally save to another file."""
    with open(input_file) as f:
        results = json.load(f)

    graded = grade_results(results)

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(graded, f, indent=2)

    return graded


def compute_statistics(graded_results: list[dict]) -> dict:
    """Compute summary statistics from graded results."""
    successful = [r for r in graded_results if r['parse_success']]

    if not successful:
        return {
            'total': len(graded_results),
            'parse_success_count': 0,
            'parse_success_rate': 0.0,
        }

    errors = [r['error'] for r in successful if r['error'] is not None]

    return {
        'total': len(graded_results),
        'parse_success_count': len(successful),
        'parse_success_rate': len(successful) / len(graded_results),
        'mean_absolute_error': sum(errors) / len(errors) if errors else None,
        'mean_guessed_temp': sum(r['parsed_temperature'] for r in successful) / len(successful),
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        graded = grade_results_file(input_file, output_file)
        stats = compute_statistics(graded)
        print(f"Statistics: {stats}")
