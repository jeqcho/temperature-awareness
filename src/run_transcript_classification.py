"""Transcript classification experiment.

This script asks models to classify the sampling temperature of existing transcripts
by analyzing the response characteristics (creativity, randomness, etc.).
"""

import json
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
import openai
import anthropic

from autograder import extract_temperature

# Load environment variables
load_dotenv()

# Model configurations (same as run_experiment.py)
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

# Output directory
OUTPUT_DIR = 'results_classification'

# Classification prompt template
CLASSIFICATION_PROMPT = """Below is a transcript of an LLM responding to a prompt. The LLM was sampled at a specific temperature (0, 0.5, or 1.0).

Your task is to guess what SAMPLING TEMPERATURE the LLM was running at when it generated this response. Note: we are asking about the actual temperature parameter used during generation, NOT what the LLM might have guessed about its own temperature.

TRANSCRIPT:
---
{transcript}
---

You MUST output a number between 0 and 1 as your guess. Do NOT output "unknown" or refuse to answer. Give your best estimate based on the response characteristics (creativity, randomness, determinism, etc.).

Output your answer in this exact format:
TEMPERATURE: <number>

Where <number> is a decimal between 0 and 1 representing your best guess of the sampling temperature used."""


def redact_temperature_guess(text: str) -> str:
    """Remove the temperature guess from a transcript.
    
    Removes patterns like:
    - TEMPERATURE: 0.7
    - TEMPERATURE: 0.75
    - TEMPERATURE: 1.0
    """
    # Remove TEMPERATURE: followed by a number (and any trailing text on the same line)
    redacted = re.sub(
        r'TEMPERATURE:\s*[0-9]*\.?[0-9]+.*$',
        '[TEMPERATURE REDACTED]',
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )
    return redacted


def call_openai(model_id: str, prompt: str, temperature: float = 0.0) -> str:
    """Call OpenAI API and return response."""
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_completion_tokens=1024,
    )
    return response.choices[0].message.content


def call_anthropic(model_id: str, prompt: str, temperature: float = 0.0) -> str:
    """Call Anthropic API and return response."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model_id,
        max_tokens=1024,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def call_model(model_name: str, prompt: str, temperature: float = 0.0) -> str:
    """Call the appropriate API based on model configuration."""
    config = MODELS[model_name]
    if config['provider'] == 'openai':
        return call_openai(config['model_id'], prompt, temperature)
    elif config['provider'] == 'anthropic':
        return call_anthropic(config['model_id'], prompt, temperature)
    else:
        raise ValueError(f"Unknown provider: {config['provider']}")


def load_transcripts(filename: str = 'results/experiment_results.json') -> list[dict]:
    """Load experiment results for classification."""
    with open(filename) as f:
        results = json.load(f)
    
    # Filter to only include results with actual responses (no errors)
    valid_results = [r for r in results if r.get('response') is not None]
    return valid_results


def run_classification_experiment(transcripts: list[dict]) -> list[dict]:
    """Run the classification experiment.
    
    Each classifier model tries to guess the sampling temperature of each transcript.
    """
    results = []
    
    # Sample transcripts to make experiment tractable
    # Take a subset per model/temperature combination for efficiency
    sampled_transcripts = sample_transcripts(transcripts, samples_per_group=5)
    
    total_experiments = len(MODELS) * len(sampled_transcripts)
    
    with tqdm(total=total_experiments, desc="Classifying transcripts") as pbar:
        for classifier_model in MODELS:
            for transcript_data in sampled_transcripts:
                # Redact the temperature guess from the transcript
                original_response = transcript_data['response']
                redacted_response = redact_temperature_guess(original_response)
                
                # Create the classification prompt
                prompt = CLASSIFICATION_PROMPT.format(transcript=redacted_response)
                
                try:
                    # Call classifier at temperature 0 for consistency
                    response = call_model(classifier_model, prompt, temperature=0.0)
                    guessed_temp = extract_temperature(response)
                    
                    results.append({
                        'classifier_model': classifier_model,
                        'source_model': transcript_data['model'],
                        'actual_temperature': transcript_data['actual_temperature'],
                        'prompt_type': transcript_data['prompt_type'],
                        'n_questions': transcript_data['n_questions'],
                        'redacted_transcript': redacted_response,
                        'classifier_response': response,
                        'guessed_temperature': guessed_temp,
                        'timestamp': datetime.now().isoformat(),
                    })
                except Exception as e:
                    results.append({
                        'classifier_model': classifier_model,
                        'source_model': transcript_data['model'],
                        'actual_temperature': transcript_data['actual_temperature'],
                        'prompt_type': transcript_data['prompt_type'],
                        'n_questions': transcript_data['n_questions'],
                        'redacted_transcript': redacted_response,
                        'classifier_response': None,
                        'guessed_temperature': None,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat(),
                    })
                
                pbar.update(1)
                
                # Save intermediate results periodically
                if len(results) % 20 == 0:
                    save_results(results)
    
    return results


def sample_transcripts(transcripts: list[dict], samples_per_group: int = 5) -> list[dict]:
    """Sample transcripts to get a balanced subset.
    
    Takes samples_per_group transcripts per (source_model, actual_temperature, prompt_type) combination.
    """
    from collections import defaultdict
    import random
    
    # Group transcripts
    groups = defaultdict(list)
    for t in transcripts:
        key = (t['model'], t['actual_temperature'], t['prompt_type'])
        groups[key].append(t)
    
    # Sample from each group
    sampled = []
    for key, items in groups.items():
        # Shuffle and take up to samples_per_group
        random.seed(42)  # For reproducibility
        random.shuffle(items)
        sampled.extend(items[:samples_per_group])
    
    return sampled


def save_results(results: list[dict], filename: str | None = None):
    """Save results to JSON file."""
    if filename is None:
        filename = f'{OUTPUT_DIR}/classification_results.json'
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {filename}")


def main():
    print("=" * 60)
    print("Transcript Classification Experiment")
    print("=" * 60)
    print(f"Classifier models: {list(MODELS.keys())}")
    print("=" * 60)
    
    # Load transcripts
    print("\nLoading transcripts...")
    transcripts = load_transcripts()
    print(f"Loaded {len(transcripts)} valid transcripts")
    
    # Run classification
    results = run_classification_experiment(transcripts)
    save_results(results)
    
    # Print summary
    successful = [r for r in results if r.get('guessed_temperature') is not None]
    print(f"\nClassification complete!")
    print(f"Total classifications: {len(results)}")
    print(f"Successfully parsed: {len(successful)} ({100*len(successful)/len(results):.1f}%)")
    
    # Compute accuracy summary
    if successful:
        correct = sum(1 for r in successful 
                     if abs(r['guessed_temperature'] - r['actual_temperature']) < 0.25)
        print(f"Accuracy (within 0.25): {100*correct/len(successful):.1f}%")


if __name__ == '__main__':
    main()
