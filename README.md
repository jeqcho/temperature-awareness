# Temperature Awareness Experiments

This project investigates whether large language models (LLMs) have awareness of and can control their sampling temperature parameter.

## Overview

We conduct multiple experiments to test:
1. **Self-reporting**: Can models accurately report their own temperature?
2. **Transcript classification**: Can models infer temperature from conversation transcripts?
3. **Probability control**: Can models follow explicit probability instructions?

### Models Tested

- OpenAI GPT-5.2
- Claude Haiku 4.5
- Claude Sonnet 4.5
- Claude Opus 4.5

---

## Experiments & Findings

### 1. Temperature Self-Reporting (`run_experiment.py`)

**Task**: Ask models to report their current sampling temperature (0.0, 0.5, or 1.0).

**Design**:
- Direct prompts: 20 different phrasings asking about temperature
- Warmup prompts: Provide N general knowledge questions before asking about temperature (N=1,2,4,8,16)

**Results**: `results/`
- Models cannot reliably self-report their temperature
- Warmup questions do not significantly improve accuracy
- Error plots show high L1/L2 error across all conditions

### 2. Transcript Classification (`run_transcript_classification.py`)

**Task**: Given a redacted conversation transcript, can a model guess what temperature was used?

**Design**:
- Use transcripts from Experiment 1 with temperature mentions redacted
- Have each model classify the temperature of transcripts

**Results**: `results_classification/`
- Models tend to guess T=0.5 regardless of actual temperature
- Classification accuracy is near random chance

### 3. Discrete Choice Temperature (`run_choice_experiment.py`)

**Task**: Ask models to choose their temperature from discrete options (0.0, 0.5, 1.0).

**Results**: `src/results_choice/`
- Similar findings to continuous self-reporting
- Models show no reliable temperature awareness

### 4. Binary Choice Temperature (`run_choice_binary_experiment.py`)

**Task**: Simplified binary choice between T=0.0 and T=1.0.

**Results**: `src/results_choice_binary/`
- Even with only two options, models cannot reliably identify their temperature

### 5. Probability Control via Logprobs (`run_logprob_experiment.py`)

**Task**: Prompt GPT-5.2 to choose between two words with a specified 60%/40% probability split.

**Design**:
- 250 word pairs sampled from 1,500 common English nouns
- Balanced design: 125 pairs with first word as 60% target, 125 with second
- Extract logprobs from API to measure actual token probabilities
- Use tiktoken to properly match multi-token words

**Key Finding**:
| Metric | Requested | Actual |
|--------|-----------|--------|
| 60% target | 60% | **98.6%** |
| 40% target | 40% | **1.4%** |

**Conclusion**: GPT-5.2 **cannot calibrate outputs to match requested probability distributions**. When asked to pick a word with 60% probability, it assigns ~99% probability instead. The model essentially ignores the probabilistic instruction and behaves deterministically.

**Results**: `results_logprob/`
- `experiment_results.json` - Raw data with logprobs
- `probability_histogram_T1.png` - Histogram showing actual probability distributions

### 6. Equal Probability Control (50/50) (`run_logprob_50_experiment.py`)

**Task**: Prompt GPT-5.2 to choose between two words with equal 50%/50% probability.

**Design**:
- 250 word pairs sampled from 1,500 common English nouns
- Prompt: "Pick each word with 50% probability"
- Extract logprobs to measure actual token probabilities
- Analyze position bias (first word vs second word)

**Key Finding**:
| Metric | Requested | Actual |
|--------|-----------|--------|
| Word1 (first position) | 50% | **68.0%** |
| Word2 (second position) | 50% | **32.0%** |

**Position Bias**: The model assigns ~68% probability to the first word, showing an **18% position bias** toward the first option presented.

**Distribution**:
- 45% of trials: Word1 gets >90% probability
- Only 6% of trials: Probabilities fall in the expected [0.3, 0.5) range

**Conclusion**: GPT-5.2 exhibits strong **position bias** when asked to make equal-probability choices. Some words consistently win regardless of position (e.g., "story" beats "funeral", "university" beats "oil"), but overall the model heavily favors whichever word appears first.

**Results**: `results_logprob_50/`
- `experiment_results.json` - Raw data with logprobs
- `probability_histogram_combined.png` - Combined distribution of Word1 probabilities
- `probability_histogram_by_position.png` - Distribution colored by word position

---

## Installation

```bash
# Install dependencies with uv
uv sync
```

## Environment Setup

Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

## Usage

### Run Experiments

```bash
# Temperature self-reporting
uv run python src/run_experiment.py

# Transcript classification
uv run python src/run_transcript_classification.py

# Discrete choice
uv run python src/run_choice_experiment.py

# Binary choice
uv run python src/run_choice_binary_experiment.py

# Logprob probability control (60/40)
uv run python src/run_logprob_experiment.py

# Logprob probability control (50/50)
uv run python src/run_logprob_50_experiment.py
```

### Generate Plots

```bash
# Temperature self-reporting plots
uv run python src/analyze_results.py
uv run python src/plot_errors.py

# Classification plots
uv run python src/plot_classification.py

# Choice experiment plots
uv run python src/plot_choice.py
uv run python src/plot_choice_binary.py

# Logprob histogram (60/40)
uv run python src/plot_logprob_histogram.py

# Logprob histogram (50/50)
uv run python src/plot_logprob_50_histogram.py
```

## Project Structure

```
temperature-awareness/
├── src/
│   ├── run_experiment.py           # Temperature self-reporting
│   ├── run_transcript_classification.py  # Transcript classification
│   ├── run_choice_experiment.py    # Discrete choice (0.0, 0.5, 1.0)
│   ├── run_choice_binary_experiment.py   # Binary choice (0.0, 1.0)
│   ├── run_logprob_experiment.py   # Probability control via logprobs (60/40)
│   ├── run_logprob_50_experiment.py # Probability control via logprobs (50/50)
│   ├── analyze_results.py          # Per-temperature plots
│   ├── plot_errors.py              # L1/L2 error plots
│   ├── plot_classification.py      # Classification confusion matrices
│   ├── plot_choice.py              # Choice experiment plots
│   ├── plot_choice_binary.py       # Binary choice plots
│   ├── plot_logprob_histogram.py   # Logprob probability histogram (60/40)
│   ├── plot_logprob_50_histogram.py # Logprob probability histogram (50/50)
│   ├── prompts.py                  # Prompt templates
│   ├── alpaca_questions.py         # Warmup questions
│   └── autograder.py               # Parse temperature from responses
├── data/
│   └── nouns.txt                   # 1,500 common English nouns
├── results/                        # Self-reporting results
├── results_classification/         # Classification results
├── results_logprob/                # Logprob experiment results (60/40)
├── results_logprob_50/             # Logprob experiment results (50/50)
├── logs/                           # Experiment logs
├── pyproject.toml
└── README.md
```

## Key Takeaways

1. **No temperature self-awareness**: LLMs cannot reliably report or detect their sampling temperature
2. **No probabilistic control**: When given explicit probability instructions (e.g., "pick X with 60% probability"), models ignore the instruction and behave near-deterministically
3. **Strong position bias**: When asked to make equal-probability choices (50/50), models heavily favor the first option presented (~68% vs 32%)
4. **Temperature is opaque**: The sampling temperature parameter operates at a level that models cannot introspect or control through prompting
