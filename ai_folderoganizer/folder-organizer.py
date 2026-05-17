#!/usr/bin/env python3
"""
Agentic Folder Organizer using Ollama
--------------------------------------
Normal mode (default):
  Agent 1 (Coordinator): Asks which folder to organize and collects special instructions.
  Agent 2 (Classifier):  Analyzes file names (and content for text/productivity files) to
                          determine the right subfolder, reusing existing ones or creating new ones.

Verbose mode  (-v / --verbose):
  Classifies files in batches of 4, shows the user what the agent intends to do, and asks
  for feedback before committing each batch. Feedback is distilled into updated rules that
  carry forward for the rest of the run. The user can switch to automatic mode at any point.

Revision mode  (-r / --revise):
  Scans the already-organized destination folder recursively, shows the user a summary of
  what is there, collects feedback on what was placed incorrectly, then uses an AI auditor
  to propose and execute a batch of corrective moves.

Usage:
  python organize_folder.py              # normal mode
  python organize_folder.py -v           # verbose / supervised mode
  python organize_folder.py -r           # revision / correction mode
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import requests

from prompts import (
    COORDINATOR_SYSTEM,
    CATEGORIZER_SYSTEM,
    FOLDER_MATCHER_SYSTEM,
    REVISION_SYSTEM,
    VERBOSE_FEEDBACK_SYSTEM,
)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:latest"  # Change to any model you have pulled, e.g. "mistral", "phi3"

# Text / productivity extensions whose content will be read for better classification
READABLE_EXTENSIONS = {
    ".txt", ".md", ".rst", ".csv", ".json", ".xml", ".yaml", ".yml",
    ".log", ".ini", ".cfg", ".toml", ".html", ".htm",
    ".doc", ".docx", ".odt", ".rtf",
    ".xls", ".xlsx", ".ods",
    ".ppt", ".pptx", ".odp",
    ".py", ".js", ".ts", ".sh", ".bat", ".rb", ".java", ".cpp", ".c", ".cs",
}

MAX_CONTENT_CHARS = 1500  # Max characters to read from a file for context
VERBOSE_BATCH_SIZE = 4    # Files shown per feedback round in verbose mode


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def ollama_chat(prompt: str, system: str = "") -> str:
    """Send a prompt to Ollama and return the response text."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Cannot reach Ollama. Make sure it is running: `ollama serve`")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Ollama request failed: {e}")
        sys.exit(1)


def get_existing_folders(root: Path) -> list[str]:
    """Return a list of existing subfolder names inside root."""
    return [
        d.name for d in root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]


def get_files(folder: Path) -> list[Path]:
    """Return all files (non-hidden) directly inside folder (non-recursive)."""
    return [f for f in folder.iterdir() if f.is_file() and not f.name.startswith(".")]


def read_file_preview(file_path: Path) -> str:
    """Attempt to read a text preview of a file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read(MAX_CONTENT_CHARS)
    except Exception:
        return ""


def safe_move(src: Path, dest_folder: Path):
    """Move src to dest_folder, renaming if a conflict exists."""
    dest_folder.mkdir(parents=True, exist_ok=True)
    dest = dest_folder / src.name
    if dest.exists():
        stem, suffix = src.stem, src.suffix
        counter = 1
        while dest.exists():
            dest = dest_folder / f"{stem}_{counter}{suffix}"
            counter += 1
    shutil.move(str(src), str(dest))
    return dest


# ─────────────────────────────────────────────
# Agent 1 – Coordinator
# ─────────────────────────────────────────────

def agent_coordinator() -> tuple[Path, Path, str]:
    """
    Interacts with the user to determine:
      - source folder to organize
      - destination root where organised subfolders will live
      - any special instructions
    Returns (source_folder, dest_root, special_instructions).
    """
    print("\n" + "═" * 60)
    print("  🗂️  Agentic Folder Organizer  (powered by Ollama)")
    print("═" * 60 + "\n")

    # ── Step 1: destination root ──────────────────────────────
    default_dest = str(Path.home() / "Organized")
    print(f"Where should organized files be stored?")
    print(f"  Press Enter to use the default: {default_dest}")
    dest_input = input("  Destination root: ").strip()
    dest_root = Path(dest_input) if dest_input else Path(default_dest)

    # ── Step 2: source folder ─────────────────────────────────
    default_source = str(Path.home() / "Downloads")
    print(f"\nWhich folder do you want to organize?")
    print(f"  Press Enter to use the default: {default_source}")
    source_input = input("  Source folder: ").strip()
    source_folder = Path(source_input) if source_input else Path(default_source)

    if not source_folder.exists():
        print(f"\n[ERROR] Folder not found: {source_folder}")
        sys.exit(1)

    # ── Step 3: special instructions via Agent 1 ─────────────
    print("\nDo you have any special instructions for organizing? (press Enter to skip)")
    raw_instructions = input("  Instructions: ").strip()

    if raw_instructions:
        # Let Agent 1 rephrase/clarify the instructions for Agent 2
        clarified = ollama_chat(
            prompt=(
                f"The user wants to organize the folder '{source_folder}' into '{dest_root}'.\n"
                f"Their special instructions are: \"{raw_instructions}\"\n\n"
                "Rewrite these instructions as clear, concise rules for a file-classification agent. "
                "Return only the rules, no preamble."
            ),
            system=COORDINATOR_SYSTEM,
        )
        special_instructions = clarified
        print(f"\n[Agent 1] Refined instructions:\n{special_instructions}\n")
    else:
        special_instructions = "No special instructions. Use sensible defaults."

    return source_folder, dest_root, special_instructions


# ─────────────────────────────────────────────
# Agent 2 – Classifier
# ─────────────────────────────────────────────

def sanitize_folder_name(raw: str) -> str:
    """Clean up a model response into a safe folder name."""
    # Take only the first line, strip quotes/punctuation and path separators
    name = raw.splitlines()[0].strip().strip("\"'`.").replace("/", "-").replace("\\", "-")
    # Collapse multiple spaces
    name = " ".join(name.split())
    return name if name else "Unsorted"


def step1_identify_category(file: Path, content_preview: str, special_instructions: str) -> str:
    """
    Step 1: Ask the model to independently categorize the file WITHOUT
    seeing any existing folders — eliminating anchoring/recency bias.
    Returns a raw category label.
    """
    prompt = f"File name: {file.name}\n"
    if content_preview:
        prompt += f"Content preview:\n{content_preview}\n"
    if special_instructions and special_instructions != "No special instructions. Use sensible defaults.":
        prompt += f"Special instructions to consider: {special_instructions}\n"
    prompt += "\nWhat is the best category label for this file?"

    return ollama_chat(prompt=prompt, system=CATEGORIZER_SYSTEM)


def step2_match_or_create(category: str, existing_folders: list[str]) -> str:
    """
    Step 2: Given the independently chosen category, ask the model whether
    any existing folder is a genuine semantic match. If not, it returns the
    new category name to use as-is.
    Returns the final folder name.
    """
    if not existing_folders:
        return category  # Nothing to compare against — use new category directly

    existing_str = "\n".join(f"- {f}" for f in existing_folders)
    prompt = (
        f"File category: {category}\n\n"
        f"Existing folders:\n{existing_str}\n\n"
        "Does one of these folders strongly match this category? "
        "If yes, return that folder name exactly. If no, return the category label."
    )

    return ollama_chat(prompt=prompt, system=FOLDER_MATCHER_SYSTEM)


def agent_classifier(
    file: Path,
    existing_folders: list[str],
    special_instructions: str,
) -> str:
    """
    Two-step classification:
      Step 1 — identify the file's category independently (no folder bias).
      Step 2 — match against existing folders only if there's a strong semantic fit.
    Returns the final subfolder name.
    """
    content_preview = ""
    if file.suffix.lower() in READABLE_EXTENSIONS:
        content_preview = read_file_preview(file)

    # Step 1: unbiased category
    raw_category = step1_identify_category(file, content_preview, special_instructions)
    category = sanitize_folder_name(raw_category)

    # Step 2: match or create
    raw_final = step2_match_or_create(category, existing_folders)
    final = sanitize_folder_name(raw_final)

    # Safety: if the model returned something empty or weirdly long, fall back
    if not final or len(final) > 60:
        final = category if category else "Unsorted"

    return final


# ─────────────────────────────────────────────
# Revision mode helpers
# ─────────────────────────────────────────────

def get_all_files_recursive(root: Path) -> list[Path]:
    """Return every file nested anywhere under root, skipping hidden items."""
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.parts)
    )


def build_file_tree_summary(root: Path, files: list[Path]) -> str:
    """
    Produce a human-readable tree grouped by immediate subfolder, e.g.:
      Invoices/   (3 files)
        invoice_jan.pdf
        invoice_feb.pdf
        receipt_amazon.png
      Music/      (2 files)
        ...
    Also returns the flat relative paths used in the AI prompt.
    """
    from collections import defaultdict
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        rel = f.relative_to(root)
        top = rel.parts[0] if len(rel.parts) > 1 else "(root)"
        groups[top].append(f)

    lines = []
    for folder in sorted(groups):
        lines.append(f"  📁 {folder}/   ({len(groups[folder])} file(s))")
        for f in groups[folder]:
            lines.append(f"       {f.relative_to(root)}")
    return "\n".join(lines)


def parse_revision_json(raw: str) -> list[dict]:
    """
    Strip any accidental markdown fences and parse the JSON array the
    revision agent returns.  Returns [] on failure.
    """
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    return []


def run_revision_mode():
    """
    Revision mode (-r):
      1. Ask user for the organized destination folder.
      2. Recursively scan it and display a grouped summary.
      3. Ask user what was placed incorrectly (free-text feedback).
      4. Send tree + feedback to REVISION_SYSTEM agent.
      5. Parse the JSON move list and execute the moves.
    """
    print("\n" + "═" * 60)
    print("  🔁  Revision Mode  (powered by Ollama)")
    print("═" * 60 + "\n")

    # ── Step 1: which organized folder to audit ───────────────
    default_dest = str(Path.home() / "Organized")
    print("Which organized folder should be audited?")
    print(f"  Press Enter to use the default: {default_dest}")
    dest_input = input("  Organized folder: ").strip()
    dest_root = Path(dest_input) if dest_input else Path(default_dest)

    if not dest_root.exists():
        print(f"\n[ERROR] Folder not found: {dest_root}")
        sys.exit(1)

    # ── Step 2: recursive scan & display ─────────────────────
    all_files = get_all_files_recursive(dest_root)
    if not all_files:
        print(f"\nNo files found under {dest_root}. Nothing to revise.")
        sys.exit(0)

    existing_folders = get_existing_folders(dest_root)

    print(f"\nFound {len(all_files)} file(s) across {len(existing_folders)} subfolder(s):\n")
    print(build_file_tree_summary(dest_root, all_files))
    print()

    # ── Step 3: collect user feedback ────────────────────────
    print("─" * 60)
    print("Describe what was organized incorrectly.")
    print("Be as specific or general as you like, e.g.:")
    print('  "All the PDFs in Music/ should be in Documents/"')
    print('  "The .py files ended up scattered — they should all go in Code/"')
    print('  "Anything with \'invoice\' in the name belongs in Finance/"')
    print()
    feedback_lines = []
    print("Enter your feedback (blank line to finish):")
    while True:
        line = input("  > ").rstrip()
        if line == "":
            break
        feedback_lines.append(line)

    if not feedback_lines:
        print("\nNo feedback provided. Exiting revision mode.")
        sys.exit(0)

    user_feedback = "\n".join(feedback_lines)

    # ── Step 4: build AI prompt & get move decisions ─────────
    # Build a compact flat file list for the prompt (relative paths)
    flat_file_list = "\n".join(
        str(f.relative_to(dest_root)).replace("\\", "/") for f in all_files
    )
    folders_list = "\n".join(f"- {f}" for f in existing_folders)

    prompt = (
        f"Current file layout (relative paths from organized root):\n"
        f"{flat_file_list}\n\n"
        f"Available subfolders:\n{folders_list}\n\n"
        f"User feedback on what is wrong:\n{user_feedback}\n\n"
        "Return the JSON move list now."
    )

    print(f"\n[Revision Agent] Analyzing {len(all_files)} files against your feedback ...\n")
    raw_response = ollama_chat(prompt=prompt, system=REVISION_SYSTEM)

    moves = parse_revision_json(raw_response)

    if not moves:
        print("[Revision Agent] No moves were identified, or the response could not be parsed.")
        print(f"Raw response:\n{raw_response}")
        sys.exit(0)

    # ── Step 5: preview & confirm ─────────────────────────────
    print(f"[Revision Agent] Proposed {len(moves)} move(s):\n")
    valid_moves = []
    for m in moves:
        file_rel = m.get("file", "").replace("\\", "/")
        target_folder = sanitize_folder_name(m.get("to", ""))
        if not file_rel or not target_folder:
            continue
        src = dest_root / Path(file_rel)
        if not src.exists():
            print(f"  ⚠️  Skipping (not found): {file_rel}")
            continue
        current_folder = Path(file_rel).parts[0] if len(Path(file_rel).parts) > 1 else "(root)"
        print(f"  {current_folder}/  →  {target_folder}/  │  {src.name}")
        valid_moves.append((src, dest_root / target_folder))

    if not valid_moves:
        print("\nNo valid moves to execute.")
        sys.exit(0)

    print()
    confirm = input(f"Apply {len(valid_moves)} move(s)? [Y/n]: ").strip().lower()
    if confirm and confirm != "y":
        print("Revision cancelled.")
        sys.exit(0)

    # ── Step 6: execute moves ─────────────────────────────────
    print()
    moved_count = 0
    for src, target_dir in valid_moves:
        print(f"  Moving  {src.relative_to(dest_root)}  →  {target_dir.name}/", end=" ")
        try:
            safe_move(src, target_dir)
            print("✓")
            moved_count += 1
        except Exception as e:
            print(f"✗  ({e})")

    # Clean up empty directories left behind
    for folder in dest_root.iterdir():
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()

    print(f"\n✅  Revision complete — {moved_count} file(s) moved.\n")


# ─────────────────────────────────────────────
# Verbose mode
# ─────────────────────────────────────────────

def distill_feedback(
    current_instructions: str,
    batch: list[tuple[str, str]],  # [(filename, proposed_folder), ...]
    user_feedback: str,
) -> str:
    """
    Send the current instructions, the batch preview, and the user's feedback
    to VERBOSE_FEEDBACK_SYSTEM and return an updated instruction string.
    """
    batch_lines = "\n".join(f"  {fname}  →  {folder}/" for fname, folder in batch)
    prompt = (
        f"Current classification rules:\n{current_instructions}\n\n"
        f"Files just classified (name → proposed folder):\n{batch_lines}\n\n"
        f"User feedback on what is wrong:\n{user_feedback}\n\n"
        "Return the updated rule set now."
    )
    return ollama_chat(prompt=prompt, system=VERBOSE_FEEDBACK_SYSTEM)


def classify_one(
    file: Path,
    dest_root: Path,
    special_instructions: str,
) -> str:
    """Run the two-step classifier for a single file; returns the folder name."""
    existing_folders = get_existing_folders(dest_root) if dest_root.exists() else []

    content_preview = ""
    if file.suffix.lower() in READABLE_EXTENSIONS:
        content_preview = read_file_preview(file)

    raw_category = step1_identify_category(file, content_preview, special_instructions)
    category = sanitize_folder_name(raw_category)

    raw_final = step2_match_or_create(category, existing_folders)
    final = sanitize_folder_name(raw_final)

    if not final or len(final) > 60:
        final = category if category else "Unsorted"
    return final


def run_verbose_mode():
    """
    Verbose mode (-v):
      • Runs the normal coordinator to gather source, destination, and initial instructions.
      • Processes files in batches of VERBOSE_BATCH_SIZE.
      • After classifying each batch, shows the user the proposed moves and asks for feedback.
      • Feedback is distilled into updated instructions carried forward for all remaining files.
      • After any batch the user can type 'auto' to switch to fully automatic mode.
      • Approved files in each batch are moved immediately before the next batch begins.
    """
    print("\n" + "═" * 60)
    print("  🔍  Verbose Mode  (powered by Ollama)")
    print("═" * 60)
    print("  Files are classified in batches. After each batch you can")
    print("  correct mistakes — feedback updates the agent going forward.")
    print("  Type  'auto'  at any feedback prompt to finish automatically.")
    print("═" * 60 + "\n")

    # ── Gather intent (same as normal mode) ──────────────────
    source_folder, dest_root, special_instructions = agent_coordinator()

    files = get_files(source_folder)
    if not files:
        print(f"\nNo files found in {source_folder}. Nothing to do.")
        sys.exit(0)

    total = len(files)
    print(f"\n📂 Source      : {source_folder}")
    print(f"📁 Destination : {dest_root}")
    print(f"📄 Files found : {total}")
    print(f"📦 Batch size  : {VERBOSE_BATCH_SIZE}\n")
    print("─" * 60)

    confirm = input("Proceed with verbose organizing? [Y/n]: ").strip().lower()
    if confirm and confirm != "y":
        print("Aborted.")
        sys.exit(0)

    results: list[tuple[str, str, Path]] = []
    auto_mode = False
    batch_num = 0
    i = 0  # global file index

    while i < total:
        batch_files = files[i: i + VERBOSE_BATCH_SIZE]
        batch_num += 1
        remaining_after = total - i - len(batch_files)

        print(f"\n{'─' * 60}")
        print(f"  Batch {batch_num}  │  files {i + 1}–{i + len(batch_files)} of {total}")
        print(f"{'─' * 60}\n")

        # ── Classify this batch ───────────────────────────────
        proposals: list[tuple[Path, str]] = []  # (file, proposed_folder)

        for file in batch_files:
            print(f"  Classifying: {file.name} ...", end=" ", flush=True)
            folder = classify_one(file, dest_root, special_instructions)
            proposals.append((file, folder))
            print(f"→  '{folder}/'")

        # ── In auto mode, commit immediately without asking ───
        if auto_mode:
            for file, folder in proposals:
                target_dir = dest_root / folder
                moved_to = safe_move(file, target_dir)
                results.append((file.name, folder, moved_to))
            i += len(batch_files)
            continue

        # ── Show batch summary and prompt for feedback ────────
        print(f"\n{'─' * 60}")
        print("  Proposed moves for this batch:\n")
        for idx, (file, folder) in enumerate(proposals, 1):
            existing = get_existing_folders(dest_root) if dest_root.exists() else []
            tag = "existing ✓" if folder in existing else "new folder ✨"
            print(f"  {idx}. {file.name:<40}  →  {folder}/  ({tag})")

        print(f"\n{'─' * 60}")
        print("  Options:")
        print("    • Press Enter  — approve all and move them")
        print("    • Type feedback — describe what's wrong (agent will learn and retry)")
        print("    • Type 'auto'  — approve this batch and switch to automatic mode")
        if remaining_after > 0:
            print(f"  ({remaining_after} file(s) remaining after this batch)")
        print()

        raw = input("  Your input: ").strip()

        # ── 'auto' — approve batch, switch mode ───────────────
        if raw.lower() == "auto":
            print("\n  Switching to automatic mode. Committing this batch and continuing...\n")
            for file, folder in proposals:
                target_dir = dest_root / folder
                moved_to = safe_move(file, target_dir)
                results.append((file.name, folder, moved_to))
            auto_mode = True
            i += len(batch_files)
            continue

        # ── Empty input — approve batch as-is ─────────────────
        if not raw:
            print("\n  ✓ Batch approved. Moving files...\n")
            for file, folder in proposals:
                target_dir = dest_root / folder
                moved_to = safe_move(file, target_dir)
                results.append((file.name, folder, moved_to))
                print(f"    Moved: {file.name}  →  {folder}/")
            i += len(batch_files)
            continue

        # ── Feedback given — distill rules, then retry batch ──
        print("\n  [Verbose Agent] Updating classification rules based on your feedback...\n")
        batch_preview = [(f.name, folder) for f, folder in proposals]
        updated_instructions = distill_feedback(special_instructions, batch_preview, raw)
        special_instructions = updated_instructions

        print(f"  Updated rules:\n")
        for line in special_instructions.splitlines():
            print(f"    {line}")
        print()
        print("  Re-classifying the batch with updated rules...\n")

        # Retry the same batch with updated instructions (do NOT advance i)
        # The loop will naturally re-process from i with new special_instructions


    # ── Final summary ─────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  ✅  Verbose organization complete!")
    print("═" * 60)

    folder_counts: dict[str, int] = {}
    for _, folder, _ in results:
        folder_counts[folder] = folder_counts.get(folder, 0) + 1

    print(f"\nFiles organized into {len(folder_counts)} folder(s) inside '{dest_root}':\n")
    for folder, count in sorted(folder_counts.items()):
        print(f"  📁 {folder:<30} {count} file(s)")

    print(f"\nTotal files moved : {len(results)}")
    if auto_mode:
        print("  (finished in automatic mode)")
    print()


# ─────────────────────────────────────────────
# Main orchestration
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Agentic Folder Organizer — uses Ollama to sort files into subfolders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python organize_folder.py          # normal automatic mode\n"
            "  python organize_folder.py -v       # verbose / supervised mode\n"
            "  python organize_folder.py -r       # revision / correction mode\n"
        ),
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help=(
            "Verbose mode: classify files in batches of 4, show proposed moves, "
            "collect feedback, then update rules before continuing."
        ),
    )
    parser.add_argument(
        "-r", "--revise",
        action="store_true",
        help="Revision mode: audit an already-organized folder and fix mistakes.",
    )
    args = parser.parse_args()

    if args.verbose and args.revise:
        print("[ERROR] -v and -r cannot be used together. Pick one mode.")
        sys.exit(1)

    if args.verbose:
        run_verbose_mode()
        return

    if args.revise:
        run_revision_mode()
        return

    # ── Normal mode ───────────────────────────────────────────
    # Agent 1: gather user intent
    source_folder, dest_root, special_instructions = agent_coordinator()

    files = get_files(source_folder)
    if not files:
        print(f"\nNo files found in {source_folder}. Nothing to do.")
        sys.exit(0)

    print(f"\n📂 Source      : {source_folder}")
    print(f"📁 Destination : {dest_root}")
    print(f"📄 Files found : {len(files)}\n")
    print("─" * 60)

    confirm = input("Proceed with organizing? [Y/n]: ").strip().lower()
    if confirm and confirm != "y":
        print("Aborted.")
        sys.exit(0)

    print()

    # ── Agent 2: classify & move each file ───────────────────
    results = []

    for i, file in enumerate(files, 1):
        existing_folders = get_existing_folders(dest_root) if dest_root.exists() else []

        print(f"[{i}/{len(files)}] {file.name}")
        print(f"         Step 1: identifying category ...", end=" ", flush=True)

        content_preview = ""
        if file.suffix.lower() in READABLE_EXTENSIONS:
            content_preview = read_file_preview(file)

        raw_category = step1_identify_category(file, content_preview, special_instructions)
        category = sanitize_folder_name(raw_category)
        print(f"'{category}'")

        print(f"         Step 2: matching to folders  ...", end=" ", flush=True)
        raw_final = step2_match_or_create(category, existing_folders)
        subfolder_name = sanitize_folder_name(raw_final)
        if not subfolder_name or len(subfolder_name) > 60:
            subfolder_name = category if category else "Unsorted"

        matched = subfolder_name in existing_folders
        tag = "reused ✓" if matched else "new folder ✨"
        print(f"→  '{subfolder_name}/'  ({tag})")
        print()

        target_dir = dest_root / subfolder_name
        moved_to = safe_move(file, target_dir)
        results.append((file.name, subfolder_name, moved_to))

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  ✅  Organization complete!")
    print("═" * 60)

    folder_counts: dict[str, int] = {}
    for _, folder, _ in results:
        folder_counts[folder] = folder_counts.get(folder, 0) + 1

    print(f"\nFiles organized into {len(folder_counts)} folder(s) inside '{dest_root}':\n")
    for folder, count in sorted(folder_counts.items()):
        print(f"  📁 {folder:<30} {count} file(s)")

    print(f"\nTotal files moved: {len(results)}")
    print()


if __name__ == "__main__":
    main()

