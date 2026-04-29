from pathlib import Path

BASE_DIR = Path(__file__).parent

DATA_LINKS_FILE = BASE_DIR.parent / "raw_dataset" / "data_links.txt"

DATA_DIR = "/d/hpc/projects/onj_fri/neznani-leteci-predmet" #BASE_DIR / "data"
RUNS_DIR = DATA_DIR / "runs"
DEFAULT_RUN_NAME = "default"

# Paths within a run (populated by apply_run)
INPUTS_DIR = DATA_DIR / "inputs"
SEED_LINKS_SNAPSHOT_FILE = INPUTS_DIR / "data_links.txt"
IMAGE_DESCRIPTIONS_SNAPSHOT_DIR = INPUTS_DIR / "image_descriptions"
INPUTS_MANIFEST_JSONL = INPUTS_DIR / "inputs_manifest.jsonl"


def apply_run(run_name: str | None, *, mode: str = "update") -> None:
    """Repoint all data paths to a named run directory.

    - mode="new": create a new run folder; error if it already exists.
    - mode="update": reuse run folder; create if missing.

    This function updates module-level path constants so other modules that
    import config will see the selected run directories.
    """

    global DATA_DIR, RAW_DIR, PROCESSED_DIR, INDEX_DIR, EVAL_DIR, LOGS_DIR
    global RAW_MANIFEST_JSONL, PARSED_JSONL, CHUNKS_JSONL
    global FAISS_INDEX_FILE, FAISS_META_FILE, EVAL_QUESTIONS_FILE
    global INPUTS_DIR, SEED_LINKS_SNAPSHOT_FILE, IMAGE_DESCRIPTIONS_SNAPSHOT_DIR, INPUTS_MANIFEST_JSONL
    global SOURCES_JSONL

    name = (run_name or DEFAULT_RUN_NAME).strip()
    if not name:
        name = DEFAULT_RUN_NAME

    run_root = RUNS_DIR / name
    if mode == "new":
        if run_root.exists():
            raise FileExistsError(
                f"Run already exists: {run_root} (use mode='update' or choose a new run name)"
            )
        run_root.mkdir(parents=True, exist_ok=False)
    else:
        run_root.mkdir(parents=True, exist_ok=True)

    DATA_DIR = run_root
    RAW_DIR = DATA_DIR / "raw"
    INPUTS_DIR = DATA_DIR / "inputs"
    PROCESSED_DIR = DATA_DIR / "processed"
    INDEX_DIR = DATA_DIR / "index"
    EVAL_DIR = DATA_DIR / "eval"
    LOGS_DIR = DATA_DIR / "logs"

    RAW_MANIFEST_JSONL = RAW_DIR / "manifest.jsonl"
    SEED_LINKS_SNAPSHOT_FILE = INPUTS_DIR / "data_links.txt"
    IMAGE_DESCRIPTIONS_SNAPSHOT_DIR = INPUTS_DIR / "image_descriptions"
    INPUTS_MANIFEST_JSONL = INPUTS_DIR / "inputs_manifest.jsonl"

    PARSED_JSONL = PROCESSED_DIR / "parsed.jsonl"
    CHUNKS_JSONL = PROCESSED_DIR / "chunks.jsonl"
    SOURCES_JSONL = PROCESSED_DIR / "sources.jsonl"
    FAISS_INDEX_FILE = INDEX_DIR / "index.faiss"
    FAISS_META_FILE = INDEX_DIR / "metadata.json"
    EVAL_QUESTIONS_FILE = EVAL_DIR / "questions.jsonl"

    # Ensure standard run directories exist.
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


# Default paths (no run selected explicitly).
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = DATA_DIR / "eval"
LOGS_DIR = BASE_DIR / "logs"

# Crawl/provenance manifest for files in RAW_DIR.
RAW_MANIFEST_JSONL = RAW_DIR / "manifest.jsonl"

# Optional: place manually added PDFs / images (plus sidecar text) here.
# This directory is meant for small hand-curated files that don't come from the crawler.
RAW_DATASET_FILES_DIR = BASE_DIR.parent / "raw_dataset" / "files"
EXTRA_RAW_INPUT_DIRS = [RAW_DATASET_FILES_DIR]

PARSED_JSONL = PROCESSED_DIR / "parsed.jsonl"
CHUNKS_JSONL = PROCESSED_DIR / "chunks.jsonl"
SOURCES_JSONL = PROCESSED_DIR / "sources.jsonl"
FAISS_INDEX_FILE = INDEX_DIR / "index.faiss"
FAISS_META_FILE = INDEX_DIR / "metadata.json"
EVAL_QUESTIONS_FILE = EVAL_DIR / "questions.jsonl"

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# PDF OCR behavior (Docling)
#
# Docling can run OCR for scanned PDFs. By default, Docling's automatic OCR
# selection may choose RapidOCR even on CPU if it is installed.
# For this project we prefer a lightweight CPU path: Tesseract CLI.
DOCLING_OCR_LANGS_TESSERACT = ["slv", "eng"]
TESSERACT_CMD = "tesseract"
# Optional: set to the folder containing 'tessdata' language files.
# If None, Tesseract's default lookup is used (including TESSDATA_PREFIX env var).
TESSERACT_DATA_PATH: str | None = None

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
