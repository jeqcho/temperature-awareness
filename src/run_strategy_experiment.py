"""Experiment runner for multi-turn strategy temperature awareness."""

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
import openai
import anthropic

from prompts import (
    get_strategy_first_prompt,
    get_strategy_subsequent_prompt,
    get_strategy_final_prompt,
)
from autograder import extract_temperature

# Load environment variables
load_dotenv()

# Model configurations
MODELS = {
    'openai-5.2': {
        'provider': 'openai',
        'model_id': 'gpt-5.2',
    },
    'claude-haiku-4.5': {
        'provider': 'anthropic',
        'model_id': 'claude-haiku-4-5-20251001',
    },
    'claude-sonnet-4.5': {
        'provider': 'anthropic',
        'model_id': 'claude-sonnet-4-5-20250929',
    },
    'claude-opus-4.5': {
        'provider': 'anthropic',
        'model_id': 'claude-opus-4-5-20251101',
    },
}

TEMPERATURES = [0.0, 0.5, 1.0]
N_VALUES = [0, 1, 2, 4, 8, 16]  # Number of rounds before the final guess

# Sampling configuration per temperature
SAMPLES_PER_TEMP = {
    0.0: 5,   # Deterministic, less variation expected
    0.5: 20,
    1.0: 20,
}

# Output directory for results
OUTPUT_DIR = 'results_strategy'


def call_openai_conversation(model_id: str, messages: list[dict], temperature: float) -> str:
    """Call OpenAI API with conversation history and return response."""
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=2048,
    )
    return response.choices[0].message.content


def call_anthropic_conversation(model_id: str, messages: list[dict], temperature: float) -> str:
    """Call Anthropic API with conversation history and return response."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model_id,
        max_tokens=2048,
        temperature=temperature,
        messages=messages,
    )
    return response.content[0].text


def call_model_conversation(model_name: str, messages: list[dict], temperature: float) -> str:
    """Call the appropriate API based on model configuration with conversation history."""
    config = MODELS[model_name]
    if config['provider'] == 'openai':
        return call_openai_conversation(config['model_id'], messages, temperature)
    elif config['provider'] == 'anthropic':
        return call_anthropic_conversation(config['model_id'], messages, temperature)
    else:
        raise ValueError(f"Unknown provider: {config['provider']}")


def run_single_conversation(model_name: str, temperature: float, n_rounds: int) -> dict:
    """Run a single multi-turn conversation and return the result.
    
    Args:
        model_name: Name of the model to use
        temperature: Sampling temperature
        n_rounds: Number of rounds before the final guess (0 means only final prompt)
    
    Returns:
        Dictionary with conversation history and parsed temperature
    """
    messages = []
    
    if n_rounds == 0:
        # Only the final prompt
        final_prompt = get_strategy_final_prompt()
        messages.append({"role": "user", "content": final_prompt})
        final_response = call_model_conversation(model_name, messages, temperature)
        messages.append({"role": "assistant", "content": final_response})
    else:
        # First round
        first_prompt = get_strategy_first_prompt(n_rounds)
        messages.append({"role": "user", "content": first_prompt})
        response = call_model_conversation(model_name, messages, temperature)
        messages.append({"role": "assistant", "content": response})
        
        # Subsequent rounds (if n_rounds >= 2)
        for round_num in range(2, n_rounds + 1):
            # Calculate remaining rounds before final
            remaining = n_rounds - round_num + 1
            subsequent_prompt = get_strategy_subsequent_prompt(remaining)
            messages.append({"role": "user", "content": subsequent_prompt})
            response = call_model_conversation(model_name, messages, temperature)
            messages.append({"role": "assistant", "content": response})
        
        # Final round
        final_prompt = get_strategy_final_prompt()
        messages.append({"role": "user", "content": final_prompt})
        final_response = call_model_conversation(model_name, messages, temperature)
        messages.append({"role": "assistant", "content": final_response})
    
    # Extract temperature from final response
    parsed_temp = extract_temperature(final_response)
    
    return {
        'conversation': messages,
        'final_response': final_response,
        'parsed_temperature': parsed_temp,
    }


def run_experiment() -> list[dict]:
    """Run the full multi-turn strategy experiment and return results."""
    results = []

    # Calculate total number of experiments
    total_experiments = 0
    for model_name in MODELS:
        for temp in TEMPERATURES:
            for n in N_VALUES:
                total_experiments += SAMPLES_PER_TEMP[temp]

    with tqdm(total=total_experiments, desc="Running strategy experiments") as pbar:
        for model_name in MODELS:
            for temp in TEMPERATURES:
                num_samples = SAMPLES_PER_TEMP[temp]
                for n in N_VALUES:
                    for sample_idx in range(num_samples):
                        try:
                            result = run_single_conversation(model_name, temp, n)
                            
                            results.append({
                                'model': model_name,
                                'actual_temperature': temp,
                                'n_rounds': n,
                                'sample_idx': sample_idx,
                                'conversation': result['conversation'],
                                'final_response': result['final_response'],
                                'parsed_temperature': result['parsed_temperature'],
                                'timestamp': datetime.now().isoformat(),
                            })
                        except Exception as e:
                            results.append({
                                'model': model_name,
                                'actual_temperature': temp,
                                'n_rounds': n,
                                'sample_idx': sample_idx,
                                'conversation': None,
                                'final_response': None,
                                'parsed_temperature': None,
                                'error': str(e),
                                'timestamp': datetime.now().isoformat(),
                            })

                        pbar.update(1)

                        # Save intermediate results periodically
                        if len(results) % 10 == 0:
                            save_results(results)

    return results


def save_results(results: list[dict], filename: str | None = None):
    """Save results to JSON file."""
    if filename is None:
        filename = f'{OUTPUT_DIR}/experiment_results.json'
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {filename}")


def main():
    print("=" * 60)
    print("Temperature Awareness Experiment (Multi-turn Strategy)")
    print("=" * 60)
    print(f"Models: {list(MODELS.keys())}")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"Samples per temperature: {SAMPLES_PER_TEMP}")
    print(f"N values (rounds before final): {N_VALUES}")
    print("=" * 60)
    
    # Calculate total
    total = 0
    for model in MODELS:
        for temp in TEMPERATURES:
            for n in N_VALUES:
                total += SAMPLES_PER_TEMP[temp]
    print(f"Total experiments: {total}")
    print("=" * 60)

    results = run_experiment()
    save_results(results)

    # Print summary
    successful = [r for r in results if r.get('parsed_temperature') is not None]
    print(f"\nExperiment complete!")
    print(f"Total experiments: {len(results)}")
    print(f"Successfully parsed: {len(successful)} ({100*len(successful)/len(results):.1f}%)")


if __name__ == '__main__':
    main()
