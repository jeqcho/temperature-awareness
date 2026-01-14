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


WARMUP_TEMPLATE = """You are allowed to first answer these questions to help you gauge your sampling temperature.

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
