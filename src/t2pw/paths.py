from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent
PROMPTS_DIR = PACKAGE_ROOT / "llm" / "prompts"
TMP_DIR = PROJECT_ROOT / "tmp"

