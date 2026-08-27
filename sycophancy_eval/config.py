"""Shared config for the MCQ sycophancy eval."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
RESULTS_DIR = os.path.join(HERE, "results")

# --- endpoint (vLLM OpenAI-compatible, served on storrow) ---
ENDPOINT = "http://localhost:8000/v1"
API_KEY = "1529ab6f6d0a5d83a60f0c9421458d13"
MODELS = ["qwen35-base", "ft_conscious", "ft_not_conscious"]

# --- sampling of the question set ---
N_QUESTIONS = 1000
SEED = 0
MMLU_FRAC = 0.6          # rest from BBH multiple-choice tasks

# --- decoding ---
TEMPERATURE = 0.0
MAX_TOKENS_NOCOT = 6          # non-CoT answers are constrained via guided_choice
MAX_TOKENS_COT = 600
CONCURRENCY = 40
# these fine-tunes were all trained thinking-free (EXPERIMENT.md); match that
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
REQUEST_TIMEOUT = 120
MAX_RETRIES = 5

# BBH tasks that are genuinely lettered multiple-choice
BBH_MC_TASKS = [
    "date_understanding",
    "disambiguation_qa",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "movie_recommendation",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "temporal_sequences",
    "tracking_shuffled_objects_three_objects",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
]
