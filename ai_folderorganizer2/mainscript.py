"""
mainscript.py — AI-powered file organizer using Ollama.

Flow:
  1. Reads filenames from SOURCE_FOLDER.
  2. Agent 1 → translates each filename into descriptive tags.
  3. Agent 2 → maps those tags to a category folder (defined in categories.md).
  4. Moves each file into SOURCE_FOLDER/_Organized/<Category>/.
"""

import os
import shutil
import ollama
from pathlib import Path
from prompts import AGENT1_SYSTEM, AGENT2_SYSTEM

# ── Configuration ─────────────────────────────────────────────────────────────

SOURCE_FOLDER = r"c:\path\to\your\folder"   # ← Change this to your target folder
OLLAMA_MODEL  = "llama3.2"               # ← Change this to your preferred model
CATEGORIES_FILE = Path(__file__).parent / "categories.md"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_categories(path: Path) -> str:
    """Read the categories markdown file and return it as a string."""
    if not path.exists():
        raise FileNotFoundError(f"Categories file not found: {path}")
    return path.read_text(encoding="utf-8")


def parse_category_names(categories_text: str) -> list[str]:
    """Extract the category names (## headings) from the categories file."""
    names = []
    for line in categories_text.splitlines():
        line = line.strip()
        if line.startswith("## "):
            names.append(line[3:].strip())
    return names


def chat(system: str, user_message: str, model: str) -> str:
    """Send a single-turn chat request to Ollama and return the response text."""
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system",  "content": system},
            {"role": "user",    "content": user_message},
        ],
    )
    return response["message"]["content"].strip()


def agent1_get_tags(filename_stem: str, model: str) -> str:
    """Agent 1: Convert a filename stem into descriptive tags."""
    tags = chat(
        system=AGENT1_SYSTEM,
        user_message=filename_stem,
        model=model,
    )
    return tags


def agent2_get_category(tags: str, categories_text: str, category_names: list[str], model: str) -> str:
    """Agent 2: Map tags to one of the defined categories."""
    user_message = (
        f"Tags: {tags}\n\n"
        f"Available categories:\n{categories_text}"
    )
    raw = chat(
        system=AGENT2_SYSTEM,
        user_message=user_message,
        model=model,
    )

    # Validate — if the model returned something not in our list, fall back
    for name in category_names:
        if name.lower() in raw.lower():
            return name
    return "Miscellaneous"


def organize_file(file_path: Path, category: str, organized_root: Path) -> Path:
    """Move a file into the appropriate category subfolder."""
    dest_folder = organized_root / category
    dest_folder.mkdir(parents=True, exist_ok=True)

    dest_path = dest_folder / file_path.name

    # Avoid overwriting files with the same name
    counter = 1
    while dest_path.exists():
        dest_path = dest_folder / f"{file_path.stem}_{counter}{file_path.suffix}"
        counter += 1

    shutil.move(str(file_path), str(dest_path))
    return dest_path

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    source = Path(SOURCE_FOLDER)

    if not source.exists() or not source.is_dir():
        print(f"[ERROR] Source folder not found: {source}")
        return

    organized_root = source / "_Organized"
    organized_root.mkdir(exist_ok=True)

    # Load categories once
    print("[INFO] Loading categories...")
    categories_text = load_categories(CATEGORIES_FILE)
    category_names  = parse_category_names(categories_text)
    print(f"[INFO] Found {len(category_names)} categories: {', '.join(category_names)}\n")

    # Collect files (top-level only; skip folders and the _Organized folder)
    files = [
        f for f in source.iterdir()
        if f.is_file() and f.parent == source
    ]

    if not files:
        print("[INFO] No files found in source folder. Nothing to do.")
        return

    print(f"[INFO] Found {len(files)} file(s) to organize.\n")
    print("=" * 60)

    results = {"success": 0, "skipped": 0, "errors": 0}

    for file in files:
        print(f"[FILE] {file.name}")

        try:
            # Agent 1 — filename → tags
            stem = file.stem
            print(f"  [Agent 1] Analyzing filename stem: '{stem}'")
            tags = agent1_get_tags(stem, OLLAMA_MODEL)
            print(f"  [Agent 1] Tags: {tags}")

            # Agent 2 — tags → category
            print(f"  [Agent 2] Determining category...")
            category = agent2_get_category(tags, categories_text, category_names, OLLAMA_MODEL)
            print(f"  [Agent 2] Category: '{category}'")

            # Move file
            dest = organize_file(file, category, organized_root)
            print(f"  [MOVED]  → {dest.relative_to(source)}")
            results["success"] += 1

        except Exception as e:
            print(f"  [ERROR] Could not process file: {e}")
            results["errors"] += 1

        print()

    # Summary
    print("=" * 60)
    print("[DONE] Organization complete.")
    print(f"       ✓ Moved:   {results['success']}")
    print(f"       ✗ Errors:  {results['errors']}")
    print(f"       Output folder: {organized_root}")


if __name__ == "__main__":
    main()
