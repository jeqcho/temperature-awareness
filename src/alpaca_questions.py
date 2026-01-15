"""Questions from Alpaca dataset for warmup prompts."""

# Diverse questions from Alpaca dataset covering different task types
# These are used to let the model "warm up" before guessing temperature
ALPACA_QUESTIONS = [
    # Creative/Writing
    "Write a short poem about the ocean.",
    "Come up with a creative name for a coffee shop.",
    "Write a one-sentence story about a robot.",

    # Reasoning/Logic
    "If a train travels 60 miles in 1 hour, how far will it travel in 2.5 hours?",
    "What comes next in this sequence: 2, 4, 8, 16, ?",
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",

    # Factual/Knowledge
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical symbol for gold?",
    "Name the largest planet in our solar system.",

    # Explanation/How-to
    "Explain how a rainbow forms in simple terms.",
    "Describe the steps to make a peanut butter sandwich.",
    "How does a refrigerator keep food cold?",

    # Classification/Analysis
    "Classify the following as fruits or vegetables: apple, carrot, banana, broccoli.",
    "What emotion does this sentence convey: 'I can't believe I won the lottery!'",

    # Generation/Brainstorming
    "List three benefits of regular exercise.",
    "Suggest two ways to reduce plastic waste.",
    "Give me three ideas for a weekend activity.",

    # Translation/Transformation
    "Rewrite this sentence in passive voice: 'The cat chased the mouse.'",
    "Summarize in one word: 'The meeting was long, boring, and unproductive.'",
]


def get_questions(n: int, offset: int = 0) -> list[str]:
    """Get n questions starting from offset (wraps around if needed)."""
    total = len(ALPACA_QUESTIONS)
    return [ALPACA_QUESTIONS[(offset + i) % total] for i in range(n)]
