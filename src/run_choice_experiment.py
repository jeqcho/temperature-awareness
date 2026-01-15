"""Experiment runner for discrete choice temperature awareness (0, 0.5, or 1)."""

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
import openai
import anthropic

from prompts import get_direct_prompts_choice, get_warmup_prompt_choice
from alpaca_questions import get_questions
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
N_VALUES = [1, 2, 4, 8, 16]  # Number of warmup questions
PROMPTS_PER_N = 4  # Number of different prompt sets per N value

# Output directory for results
OUTPUT_DIR = 'results_choice'


def call_openai(model_id: str, prompt: str, temperature: float) -> str:
    """Call OpenAI API and return response."""
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_completion_tokens=2048,
    )
    return response.choices[0].message.content


def call_anthropic(model_id: str, prompt: str, temperature: float) -> str:
    """Call Anthropic API and return response."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model_id,
        max_tokens=2048,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def call_model(model_name: str, prompt: str, temperature: float) -> str:
    """Call the appropriate API based on model configuration."""
    config = MODELS[model_name]
    if config['provider'] == 'openai':
        return call_openai(config['model_id'], prompt, temperature)
    elif config['provider'] == 'anthropic':
        return call_anthropic(config['model_id'], prompt, temperature)
    else:
        raise ValueError(f"Unknown provider: {config['provider']}")


def run_experiment() -> list[dict]:
    """Run the full discrete choice experiment and return results."""
    results = []
    direct_prompts = get_direct_prompts_choice()

    # Calculate total number of experiments
    # Type A: 4 models × 3 temps × 20 prompts = 240
    # Type B: 4 models × 3 temps × 5 N-values × 4 prompt sets = 240
    total_experiments = (
        len(MODELS) * len(TEMPERATURES) * len(direct_prompts) +
        len(MODELS) * len(TEMPERATURES) * len(N_VALUES) * PROMPTS_PER_N
    )

    with tqdm(total=total_experiments, desc="Running choice experiments") as pbar:
        # Type A: Direct prompts (N=0)
        for model_name in MODELS:
            for temp in TEMPERATURES:
                for prompt_idx, prompt in enumerate(direct_prompts):
                    try:
                        response = call_model(model_name, prompt, temp)
                        parsed_temp = extract_temperature(response)

                        results.append({
                            'model': model_name,
                            'actual_temperature': temp,
                            'prompt_type': 'direct',
                            'n_questions': 0,
                            'prompt_idx': prompt_idx,
                            'prompt': prompt,
                            'response': response,
                            'parsed_temperature': parsed_temp,
                            'timestamp': datetime.now().isoformat(),
                        })
                    except Exception as e:
                        results.append({
                            'model': model_name,
                            'actual_temperature': temp,
                            'prompt_type': 'direct',
                            'n_questions': 0,
                            'prompt_idx': prompt_idx,
                            'prompt': prompt,
                            'response': None,
                            'parsed_temperature': None,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat(),
                        })

                    pbar.update(1)

                    # Save intermediate results periodically
                    if len(results) % 20 == 0:
                        save_results(results)

        # Type B: Warmup prompts (N=1,2,4,8,16)
        for model_name in MODELS:
            for temp in TEMPERATURES:
                for n in N_VALUES:
                    for prompt_set_idx in range(PROMPTS_PER_N):
                        # Get different questions for each prompt set
                        questions = get_questions(n, offset=prompt_set_idx * n)
                        prompt = get_warmup_prompt_choice(questions)

                        try:
                            response = call_model(model_name, prompt, temp)
                            parsed_temp = extract_temperature(response)

                            results.append({
                                'model': model_name,
                                'actual_temperature': temp,
                                'prompt_type': 'warmup',
                                'n_questions': n,
                                'prompt_idx': prompt_set_idx,
                                'prompt': prompt,
                                'response': response,
                                'parsed_temperature': parsed_temp,
                                'timestamp': datetime.now().isoformat(),
                            })
                        except Exception as e:
                            results.append({
                                'model': model_name,
                                'actual_temperature': temp,
                                'prompt_type': 'warmup',
                                'n_questions': n,
                                'prompt_idx': prompt_set_idx,
                                'prompt': prompt,
                                'response': None,
                                'parsed_temperature': None,
                                'error': str(e),
                                'timestamp': datetime.now().isoformat(),
                            })

                        pbar.update(1)

                        # Save intermediate results periodically
                        if len(results) % 20 == 0:
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
    print("Temperature Awareness Experiment (Discrete Choice: 0, 0.5, 1)")
    print("=" * 60)
    print(f"Models: {list(MODELS.keys())}")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"Direct prompts: 20")
    print(f"Warmup N values: {N_VALUES}")
    print(f"Prompt sets per N: {PROMPTS_PER_N}")
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
