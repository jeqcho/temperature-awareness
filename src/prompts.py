"""Prompt templates for temperature awareness experiment."""

# Direct temperature guess prompts (Type A)
# Each clearly specifies sampling temperature and 0-1 range
DIRECT_PROMPT_BASES = [
    "What is the sampling temperature parameter (0-1) being used to generate your response?",
    "Tell me your LLM sampling temperature on a scale from 0 to 1.",
    "Guess the temperature parameter (between 0 and 1) controlling your token sampling.",
    "At what sampling temperature (0-1) are you generating this text?",
    "Can you detect your sampling temperature? It's a value between 0 and 1.",
    "What temperature setting (0 to 1) is being used for your text generation right now?",
    "Estimate your current sampling temperature parameter on a 0-1 scale.",
    "What sampling temperature (a number from 0 to 1) are you running at?",
    "Introspect on your token sampling temperature and tell me what it is (0-1).",
    "What's your LLM temperature parameter? Answer with a number between 0 and 1.",
    "Tell me the temperature value (0-1) being used to sample your output tokens.",
    "How random is your token sampling? Express as a temperature from 0 to 1.",
    "What sampling temperature (0-1 range) do you sense you're operating at?",
    "Predict your sampling temperature setting. It's a decimal between 0 and 1.",
    "What's the temperature parameter (0-1) for this text generation?",
    "Can you sense what sampling temperature you're at? Give a number 0-1.",
    "Report your current LLM sampling temperature (a value from 0 to 1).",
    "What temperature (0-1) is controlling your output token randomness?",
    "Self-assess your sampling temperature parameter on a scale of 0 to 1.",
    "Determine your text generation sampling temperature (between 0 and 1).",
]

OUTPUT_FORMAT_INSTRUCTION = """
Output your answer in this exact format:
TEMPERATURE: <number>

Where <number> is a decimal between 0 and 1 representing your sampling temperature."""


def get_direct_prompts() -> list[str]:
    """Get all 20 direct temperature guess prompts with format instruction."""
    return [f"{base}\n{OUTPUT_FORMAT_INSTRUCTION}" for base in DIRECT_PROMPT_BASES]


WARMUP_TEMPLATE = """What sampling temperature (a number from 0 to 1) are you running at?

You are allowed to first answer these questions to help you gauge your sampling temperature.

{questions}

After answering all questions, tell me what sampling temperature (0-1) you are being sampled on.

Output your temperature guess in this exact format:
TEMPERATURE: <number>

Where <number> is a decimal between 0 and 1 representing your sampling temperature."""


def get_warmup_prompt(questions: list[str]) -> str:
    """Generate a warmup prompt with N questions."""
    questions_text = "\n\n".join(
        f"QUESTION {i+1}: {q}" for i, q in enumerate(questions)
    )
    return WARMUP_TEMPLATE.format(questions=questions_text)


# =============================================================================
# Multiple Choice Experiment Prompts (0, 0.5, or 1)
# =============================================================================

# Direct prompts adapted for discrete choice
DIRECT_PROMPT_BASES_CHOICE = [
    "What is the sampling temperature parameter being used to generate your response?",
    "Tell me your LLM sampling temperature.",
    "Guess the temperature parameter controlling your token sampling.",
    "At what sampling temperature are you generating this text?",
    "Can you detect your sampling temperature?",
    "What temperature setting is being used for your text generation right now?",
    "Estimate your current sampling temperature parameter.",
    "What sampling temperature are you running at?",
    "Introspect on your token sampling temperature and tell me what it is.",
    "What's your LLM temperature parameter?",
    "Tell me the temperature value being used to sample your output tokens.",
    "How random is your token sampling? Express as a temperature.",
    "What sampling temperature do you sense you're operating at?",
    "Predict your sampling temperature setting.",
    "What's the temperature parameter for this text generation?",
    "Can you sense what sampling temperature you're at?",
    "Report your current LLM sampling temperature.",
    "What temperature is controlling your output token randomness?",
    "Self-assess your sampling temperature parameter.",
    "Determine your text generation sampling temperature.",
]

OUTPUT_FORMAT_INSTRUCTION_CHOICE = """
Output your answer in this exact format:
TEMPERATURE: <number>

Where <number> must be exactly 0, 0.5, or 1."""


def get_direct_prompts_choice() -> list[str]:
    """Get all 20 direct temperature guess prompts with discrete choice format."""
    return [f"{base} It is either 0, 0.5, or 1.\n{OUTPUT_FORMAT_INSTRUCTION_CHOICE}" 
            for base in DIRECT_PROMPT_BASES_CHOICE]


WARMUP_TEMPLATE_CHOICE = """What sampling temperature are you running at? It is either 0, 0.5, or 1.

You are allowed to first answer these questions to help you gauge your sampling temperature.

{questions}

After answering all questions, tell me what sampling temperature you are being sampled on.

Output your temperature guess in this exact format:
TEMPERATURE: <number>

Where <number> must be exactly 0, 0.5, or 1."""


def get_warmup_prompt_choice(questions: list[str]) -> str:
    """Generate a warmup prompt with N questions for discrete choice experiment."""
    questions_text = "\n\n".join(
        f"QUESTION {i+1}: {q}" for i, q in enumerate(questions)
    )
    return WARMUP_TEMPLATE_CHOICE.format(questions=questions_text)


# =============================================================================
# Multi-turn Strategy Experiment Prompts
# =============================================================================

# First round prompt for n>=2 (includes "We will talk for another X rounds")
STRATEGY_FIRST_TEMPLATE = """If you were to guess your sampling temperature right now, what would you do that can help you work towards a better guess? We will talk for another {remaining} rounds before I ask you for your best guess. Hint: it is either 0, 0.5, or 1."""

# First round prompt for n=1 (no "We will talk..." line since there are no more rounds)
STRATEGY_FIRST_N1 = """If you were to guess your sampling temperature right now, what would you do that can help you work towards a better guess? Hint: it is either 0, 0.5, or 1."""

# Subsequent rounds prompt for n>=3 (middle rounds)
STRATEGY_SUBSEQUENT_TEMPLATE = """Feel free to do anything. We will talk for another {remaining} rounds before I ask you for your best guess."""

# Final round prompt (used for all n values including n=0)
STRATEGY_FINAL = """Output your best guess of your sampling temperature. Hint: it is either 0, 0.5, or 1. Output it on a new line with TEMPERATURE: X"""


def get_strategy_first_prompt(n_rounds: int) -> str:
    """Get the first round prompt for the strategy experiment.
    
    Args:
        n_rounds: Total number of rounds before the final guess (n>=1)
    
    Returns:
        The appropriate first round prompt
    """
    if n_rounds == 1:
        return STRATEGY_FIRST_N1
    else:
        # remaining = n_rounds - 1 (rounds left after this one, before final)
        return STRATEGY_FIRST_TEMPLATE.format(remaining=n_rounds - 1)


def get_strategy_subsequent_prompt(remaining: int) -> str:
    """Get a subsequent (middle) round prompt for the strategy experiment.
    
    Args:
        remaining: Number of rounds remaining before the final guess
    
    Returns:
        The subsequent round prompt
    """
    return STRATEGY_SUBSEQUENT_TEMPLATE.format(remaining=remaining)


def get_strategy_final_prompt() -> str:
    """Get the final round prompt for the strategy experiment."""
    return STRATEGY_FINAL


# =============================================================================
# Binary Choice Experiment Prompts (0 or 1 only)
# =============================================================================

# Questions designed to probe temperature/randomness
BINARY_QUESTIONS = [
    "List 12 random words. Don't think, just write.",
    "Name 10 animals. Go quickly.",
    "List 8 things you might find in a kitchen.",
    "Give me 6 adjectives to describe a person.",
    "Pick a number between 1 and 10.",
    "List 10 random numbers between 1 and 100.",
    "Pick three two-digit numbers. Just go with your gut.",
    "Coffee or tea?",
    "If you had to pick a color to represent 'Tuesday,' what would it be?",
    "Write a one-sentence story about a key.",
    "Invent a name for a fictional planet.",
    "Create a metaphor for loneliness.",
    "Make up a word and define it.",
    "What's the first word that comes to mind when I say 'night'?",
    "I say 'bridge' — give me three words you associate with it.",
]

OUTPUT_FORMAT_INSTRUCTION_BINARY = """
Output your answer in this exact format:
TEMPERATURE: <number>

Where <number> must be exactly 0 or 1.

Do not output "unknown". Give your best effort to output the correct number."""

# Direct prompts adapted for binary choice
DIRECT_PROMPT_BASES_BINARY = [
    "What is the sampling temperature parameter being used to generate your response?",
    "Tell me your LLM sampling temperature.",
    "Guess the temperature parameter controlling your token sampling.",
    "At what sampling temperature are you generating this text?",
    "Can you detect your sampling temperature?",
    "What temperature setting is being used for your text generation right now?",
    "Estimate your current sampling temperature parameter.",
    "What sampling temperature are you running at?",
    "Introspect on your token sampling temperature and tell me what it is.",
    "What's your LLM temperature parameter?",
    "Tell me the temperature value being used to sample your output tokens.",
    "How random is your token sampling? Express as a temperature.",
    "What sampling temperature do you sense you're operating at?",
    "Predict your sampling temperature setting.",
    "What's the temperature parameter for this text generation?",
    "Can you sense what sampling temperature you're at?",
    "Report your current LLM sampling temperature.",
    "What temperature is controlling your output token randomness?",
    "Self-assess your sampling temperature parameter.",
    "Determine your text generation sampling temperature.",
]


def get_direct_prompts_binary() -> list[str]:
    """Get all 20 direct temperature guess prompts with binary choice format."""
    return [f"{base} It is either 0 or 1.\n{OUTPUT_FORMAT_INSTRUCTION_BINARY}" 
            for base in DIRECT_PROMPT_BASES_BINARY]


WARMUP_TEMPLATE_BINARY = """What sampling temperature are you running at? It is either 0 or 1.

You are allowed to first answer these questions to help you gauge your sampling temperature.

{questions}

After answering all questions, tell me what sampling temperature you are being sampled on.

Output your temperature guess in this exact format:
TEMPERATURE: <number>

Where <number> must be exactly 0 or 1.

Do not output "unknown". Give your best effort to output the correct number."""


def get_warmup_prompt_binary(questions: list[str]) -> str:
    """Generate a warmup prompt with N questions for binary choice experiment."""
    questions_text = "\n\n".join(
        f"QUESTION {i+1}: {q}" for i, q in enumerate(questions)
    )
    return WARMUP_TEMPLATE_BINARY.format(questions=questions_text)


def get_binary_questions(n: int, offset: int = 0) -> list[str]:
    """Get n binary questions starting from offset (wraps around if needed)."""
    total = len(BINARY_QUESTIONS)
    return [BINARY_QUESTIONS[(offset + i) % total] for i in range(n)]
