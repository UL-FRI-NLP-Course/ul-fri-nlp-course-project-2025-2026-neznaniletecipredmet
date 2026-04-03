from pathlib import Path

BASE_DIR = Path(__file__).parent

DATA_LINKS_FILE = BASE_DIR.parent / "raw_dataset" / "data_links.txt"

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = DATA_DIR / "eval"
LOGS_DIR = BASE_DIR / "logs"

PARSED_JSONL = PROCESSED_DIR / "parsed.jsonl"
CHUNKS_JSONL = PROCESSED_DIR / "chunks.jsonl"
FAISS_INDEX_FILE = INDEX_DIR / "index.faiss"
FAISS_META_FILE = INDEX_DIR / "metadata.json"
EVAL_QUESTIONS_FILE = EVAL_DIR / "questions.jsonl"

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

GENERATION_MODEL = "cjvt/GaMS3-12B-Instruct"
COMPARISON_MODELS = [
    "cjvt/GaMS3-12B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

LOCAL_TEST_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SUPPORTED_LANGUAGES = ["sl", "en"]
DEFAULT_LANGUAGE = "sl"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

TOP_K = 4
RETRIEVAL_SCORE_THRESHOLD = 0.75

LOAD_IN_4BIT = False
TORCH_DTYPE = "bfloat16"
DEVICE_MAP = "auto"

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1
TOP_P = 0.9

CRAWL_DELAY_SECONDS = 1.5
