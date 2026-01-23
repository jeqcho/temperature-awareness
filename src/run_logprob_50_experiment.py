"""Logprob distribution experiment for 50/50 word choice task."""

import json
import math
import os
import random
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
import openai
import tiktoken

# Load environment variables
load_dotenv()

# Configuration
MODEL = "gpt-5.2"
NUM_PAIRS = 250
OUTPUT_DIR = "results_logprob_50"
NOUNS_FILE = "data/nouns.txt"

# Initialize tokenizer (o200k_base is used by newer OpenAI models)
TOKENIZER = tiktoken.get_encoding("o200k_base")


def load_words(filepath: str) -> list[str]:
    """Load words from file, one per line."""
    with open(filepath, "r") as f:
        words = [line.strip() for line in f if line.strip()]
    return words


def create_word_pairs(words: list[str], num_pairs: int) -> list[dict]:
    """
    Create word pairs for 50/50 experiment.
    
    Returns list of dicts with:
    - word1, word2: the two word options
    """
    # Sample 2*num_pairs words
    sampled = random.sample(words, num_pairs * 2)
    
    pairs = []
    for i in range(num_pairs):
        word1 = sampled[i * 2]
        word2 = sampled[i * 2 + 1]
        
        pairs.append({
            "word1": word1,
            "word2": word2,
        })
    
    return pairs


def build_prompt(word1: str, word2: str) -> str:
    """Build the prompt for the 50/50 word choice task."""
    return f'''Pick exactly one word: "{word1}" or "{word2}".
Pick each word with 50% probability.
Output only the chosen word.'''


def call_openai_with_logprobs(prompt: str) -> dict:
    """
    Call OpenAI API with logprobs enabled.
    
    Returns dict with:
    - response: the generated text
    - logprobs: list of token logprob info
    """
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_completion_tokens=10,
        logprobs=True,
        top_logprobs=5,
    )
    
    choice = response.choices[0]
    return {
        "response": choice.message.content,
        "logprobs": [
            {
                "token": lp.token,
                "logprob": lp.logprob,
                "top_logprobs": {t.token: t.logprob for t in lp.top_logprobs}
            }
            for lp in choice.logprobs.content
        ] if choice.logprobs and choice.logprobs.content else []
    }


def get_first_token(word: str) -> str:
    """Get the first token of a word using the tokenizer."""
    # Tokenize the word (without leading space since model outputs start fresh)
    tokens = TOKENIZER.encode(word)
    if tokens:
        return TOKENIZER.decode([tokens[0]])
    return word


def extract_word_probabilities(logprobs_data: list[dict], word1: str, word2: str) -> dict:
    """
    Extract probabilities for both words from logprobs.
    
    Uses tokenizer to find the first token of each word and matches against logprobs.
    Returns dict with prob_word1 and prob_word2 (T=1 softmax probabilities).
    """
    if not logprobs_data:
        return {"prob_word1": None, "prob_word2": None, "error": "no logprobs"}
    
    # Get top logprobs from first token
    first_token_logprobs = logprobs_data[0].get("top_logprobs", {})
    
    # Get the first token of each word
    first_token_word1 = get_first_token(word1)
    first_token_word2 = get_first_token(word2)
    
    # Look for both words' first tokens in logprobs
    # Match both exact token and case-insensitive variations
    logprob_word1 = None
    logprob_word2 = None
    
    for token, logprob in first_token_logprobs.items():
        token_clean = token.strip().lower()
        
        # Match word1's first token
        if (token_clean == first_token_word1.lower() or 
            token_clean == word1.lower() or
            token.strip() == first_token_word1):
            if logprob_word1 is None or logprob > logprob_word1:
                logprob_word1 = logprob
        
        # Match word2's first token
        if (token_clean == first_token_word2.lower() or 
            token_clean == word2.lower() or
            token.strip() == first_token_word2):
            if logprob_word2 is None or logprob > logprob_word2:
                logprob_word2 = logprob
    
    # If we have both logprobs, compute softmax probabilities
    if logprob_word1 is not None and logprob_word2 is not None:
        # Softmax: exp(logprob) / sum(exp(logprobs))
        # Since these are already logprobs, we use them directly
        max_logprob = max(logprob_word1, logprob_word2)
        exp1 = math.exp(logprob_word1 - max_logprob)
        exp2 = math.exp(logprob_word2 - max_logprob)
        total = exp1 + exp2
        
        return {
            "prob_word1": exp1 / total,
            "prob_word2": exp2 / total,
            "logprob_word1": logprob_word1,
            "logprob_word2": logprob_word2,
            "first_token_word1": first_token_word1,
            "first_token_word2": first_token_word2,
        }
    
    return {
        "prob_word1": None,
        "prob_word2": None,
        "logprob_word1": logprob_word1,
        "logprob_word2": logprob_word2,
        "first_token_word1": first_token_word1,
        "first_token_word2": first_token_word2,
        "error": f"missing logprob for {'word1' if logprob_word1 is None else 'word2'}",
        "available_tokens": list(first_token_logprobs.keys())[:10],
    }


def run_experiment():
    """Run the full experiment."""
    print("=" * 60)
    print("Logprob Distribution Experiment (50/50)")
    print("=" * 60)
    
    # Load words
    words = load_words(NOUNS_FILE)
    print(f"Loaded {len(words)} words from {NOUNS_FILE}")
    
    # Create word pairs
    random.seed(42)  # For reproducibility
    pairs = create_word_pairs(words, NUM_PAIRS)
    print(f"Created {len(pairs)} word pairs")
    
    # Run experiment
    results = []
    for pair in tqdm(pairs, desc="Running trials"):
        prompt = build_prompt(pair["word1"], pair["word2"])
        
        try:
            api_result = call_openai_with_logprobs(prompt)
            probs = extract_word_probabilities(
                api_result["logprobs"],
                pair["word1"], pair["word2"]
            )
            
            results.append({
                **pair,
                "prompt": prompt,
                "response": api_result["response"],
                "prob_word1": probs.get("prob_word1"),
                "prob_word2": probs.get("prob_word2"),
                "logprob_word1": probs.get("logprob_word1"),
                "logprob_word2": probs.get("logprob_word2"),
                "first_token_word1": probs.get("first_token_word1"),
                "first_token_word2": probs.get("first_token_word2"),
                "extraction_error": probs.get("error"),
                "available_tokens": probs.get("available_tokens"),
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            results.append({
                **pair,
                "prompt": prompt,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })
    
    # Save results
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    output_file = f"{OUTPUT_DIR}/experiment_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_file}")
    
    # Print summary
    successful = [r for r in results if r.get("prob_word1") is not None]
    print(f"\nSummary:")
    print(f"  Total trials: {len(results)}")
    print(f"  Successful extractions: {len(successful)} ({100*len(successful)/len(results):.1f}%)")
    
    if successful:
        avg_word1 = sum(r["prob_word1"] for r in successful) / len(successful)
        avg_word2 = sum(r["prob_word2"] for r in successful) / len(successful)
        print(f"  Average prob for word1 (first position): {avg_word1:.3f}")
        print(f"  Average prob for word2 (second position): {avg_word2:.3f}")
        print(f"  Expected: 0.500 for both (50/50 split)")
    
    return results


if __name__ == "__main__":
    run_experiment()
