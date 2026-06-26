from pathlib import Path


PYTHON_DATASET = "HuggingFaceTB/smollm-corpus"
PYTHON_CONFIG = "python-edu"
WEB_DATASET = "HuggingFaceTB/smollm-corpus"
WEB_CONFIG = "fineweb-edu-dedup"
MATH_DATASET = "HuggingFaceTB/finemath"
MATH_CONFIG = "finemath-4plus"

MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
MODEL_PATH = Path("lid.176.bin")
SWH_CONTENT_URL = "https://softwareheritage.s3.amazonaws.com/content"

SOURCE_PYTHON = "python_edu"
SOURCE_WEB = "fineweb_edu_dedup"
SOURCE_MATH = "finemath_4plus"
SOURCES = (SOURCE_PYTHON, SOURCE_WEB, SOURCE_MATH)

CJK_THRESHOLD = 0.0
MIN_MIXED_TEXT_CHARS = 30
EN_CONF_MIN = 0.55
MIN_TEXT_CHARS = 30
NEAR_DEDUP_THRESHOLD = 0.60
MINHASH_NUM_PERM = 128
SHINGLE_SIZE = 7
DEFAULT_PYTHON_DEDUP_BUFFER_DOCS = 50_000
DEFAULT_SHARD_TOKENS = 50_000_000
DEFAULT_VAL_FRACTION = 0.005
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
