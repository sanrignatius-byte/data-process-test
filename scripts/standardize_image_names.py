#!/usr/bin/env python3
"""
Standalone script to standardize image names in existing MinerU output.

This script can be used to:
1. Rename hash-based image filenames to readable format
2. Update structure.json with new paths
3. Update triplets.jsonl with new paths

Usage:
    python scripts/standardize_image_names.py --input data/mineru_output
    python scripts/standardize_image_names.py --input data/mineru_output --update-triplets data/queries_output/triplets_v2.jsonl
    python scripts/standardize_image_names.py --input data/mineru_output --dry-run
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re


# Image name standardization mapping for element types
IMAGE_TYPE_PREFIX = {
    "figure": "fig",
    "table": "tbl",
    "infographic": "info",
    "chart": "chart",
    "diagram": "diag",
}


def is_hash_filename(filename: str) -> bool:
    """Check if filename appears to be a hash (e.g., SHA256, MD5, SHA1)."""
    name = Path(filename).stem
    # SHA256 produces 64 hex characters
    if len(name) == 64 and all(c in '0123456789abcdef' for c in name.lower()):
        return True
    # Also check for shorter hashes (MD5 = 32, SHA1 = 40)
    if len(name) in [32, 40] and all(c in '0123456789abcdef' for c in name.lower()):
        return True
    return False


def find_image_files(base_dir: Path) -> List[Path]:
    """Find all image files in the output directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    images = []
    for ext in image_extensions:
        images.extend(base_dir.rglob(f"*{ext}"))
        images.extend(base_dir.rglob(f"*{ext.upper()}"))
    return images


def extract_doc_id_from_path(image_path: Path, base_dir: Path) -> Optional[str]:
    """Extract document ID from image path."""
    try:
        rel_path = image_path.relative_to(base_dir)
        parts = rel_path.parts
        # Typically: doc_id/doc_id/hybrid_auto/images/hash.jpg
        if len(parts) >= 1:
            return parts[0]
    except ValueError:
        pass
    return None


def extract_page_from_structure(
    structure_path: Path,
    image_name: str
) -> Tuple[int, str]:
    """Extract page index and element type from structure.json."""
    if not structure_path.exists():
        return 0, "figure"

    try:
        with open(structure_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        elements = data.get("elements", [])
        for elem in elements:
            img_path = elem.get("image_path", "")
            if image_name in img_path or img_path.endswith(image_name):
                return elem.get("page_idx", 0), elem.get("type", "figure")

    except Exception:
        pass

    return 0, "figure"


def standardize_images(
    base_dir: Path,
    dry_run: bool = False
) -> Dict[str, str]:
    """
    Standardize all hash-based image names in the directory.

    Returns:
        Mapping from old paths to new paths
    """
    rename_map: Dict[str, str] = {}
    images = find_image_files(base_dir)

    # Group images by document
    doc_images: Dict[str, List[Path]] = {}
    for img in images:
        doc_id = extract_doc_id_from_path(img, base_dir)
        if doc_id:
            doc_images.setdefault(doc_id, []).append(img)

    for doc_id, doc_image_list in doc_images.items():
        # Find structure.json for this doc
        structure_path = base_dir / doc_id / "structure.json"

        # Track counters per (page, type)
        counters: Dict[Tuple[int, str], int] = {}

        for img_path in sorted(doc_image_list):
            if not is_hash_filename(img_path.name):
                continue

            # Get page and type from structure
            page_idx, elem_type = extract_page_from_structure(
                structure_path, img_path.name
            )

            # Get type prefix
            type_prefix = IMAGE_TYPE_PREFIX.get(elem_type, elem_type[:3])

            # Get counter
            counter_key = (page_idx, elem_type)
            counter = counters.get(counter_key, 0)

            # Generate new name
            suffix = img_path.suffix.lower()
            new_name = f"{doc_id}_page{page_idx}_{type_prefix}{counter}{suffix}"
            new_path = img_path.parent / new_name

            # Handle conflicts
            conflict_counter = 0
            while new_path.exists() and new_path != img_path:
                conflict_counter += 1
                new_name = f"{doc_id}_page{page_idx}_{type_prefix}{counter}_{conflict_counter}{suffix}"
                new_path = img_path.parent / new_name

            if img_path != new_path:
                print(f"  {img_path.name} -> {new_name}")
                rename_map[str(img_path)] = str(new_path)

                if not dry_run:
                    shutil.move(str(img_path), str(new_path))

            counters[counter_key] = counter + 1

    return rename_map


def update_structure_json(
    base_dir: Path,
    rename_map: Dict[str, str],
    dry_run: bool = False
) -> int:
    """Update structure.json files with new image paths."""
    updated_count = 0

    for structure_file in base_dir.rglob("structure.json"):
        try:
            with open(structure_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            modified = False
            elements = data.get("elements", [])

            for elem in elements:
                img_path = elem.get("image_path")
                if not img_path:
                    continue

                # Check if this path needs updating
                for old_path, new_path in rename_map.items():
                    old_name = Path(old_path).name
                    new_name = Path(new_path).name

                    if old_name in img_path:
                        elem["image_path"] = img_path.replace(old_name, new_name)
                        # Update content too
                        if "content" in elem and old_name in elem["content"]:
                            elem["content"] = elem["content"].replace(old_name, new_name)
                        # Store original in metadata
                        if "metadata" not in elem:
                            elem["metadata"] = {}
                        elem["metadata"]["original_hash_name"] = old_name
                        elem["metadata"]["standardized_name"] = new_name
                        modified = True
                        break

            if modified:
                updated_count += 1
                if not dry_run:
                    with open(structure_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"  Updated: {structure_file}")

        except Exception as e:
            print(f"  Error processing {structure_file}: {e}")

    return updated_count


def update_triplets_jsonl(
    triplets_path: Path,
    rename_map: Dict[str, str],
    dry_run: bool = False
) -> int:
    """Update triplets.jsonl with new image paths."""
    if not triplets_path.exists():
        print(f"Triplets file not found: {triplets_path}")
        return 0

    # Build name-only mapping for easier matching
    name_map = {
        Path(old).name: Path(new).name
        for old, new in rename_map.items()
    }

    updated_count = 0
    updated_lines = []

    try:
        with open(triplets_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    updated_lines.append(line)
                    continue

                data = json.loads(line)
                modified = False

                # Update positive
                if "positive" in data and data["positive"].get("image_path"):
                    img_path = data["positive"]["image_path"]
                    for old_name, new_name in name_map.items():
                        if old_name in img_path:
                            data["positive"]["image_path"] = img_path.replace(old_name, new_name)
                            modified = True
                            break

                # Update negatives
                if "negatives" in data:
                    for neg in data["negatives"]:
                        if neg.get("image_path"):
                            img_path = neg["image_path"]
                            for old_name, new_name in name_map.items():
                                if old_name in img_path:
                                    neg["image_path"] = img_path.replace(old_name, new_name)
                                    modified = True
                                    break

                if modified:
                    updated_count += 1

                updated_lines.append(json.dumps(data, ensure_ascii=False))

        if not dry_run and updated_count > 0:
            with open(triplets_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(updated_lines))
            print(f"  Updated {updated_count} triplets in {triplets_path}")

    except Exception as e:
        print(f"Error updating triplets: {e}")

    return updated_count


def update_markdown_files(
    base_dir: Path,
    rename_map: Dict[str, str],
    dry_run: bool = False
) -> int:
    """Update markdown files with new image references."""
    updated_count = 0
    name_map = {
        Path(old).name: Path(new).name
        for old, new in rename_map.items()
    }

    for md_file in base_dir.rglob("*.md"):
        try:
            content = md_file.read_text(encoding='utf-8')
            modified = False

            for old_name, new_name in name_map.items():
                if old_name in content:
                    content = content.replace(old_name, new_name)
                    modified = True

            if modified:
                updated_count += 1
                if not dry_run:
                    md_file.write_text(content, encoding='utf-8')
                print(f"  Updated: {md_file}")

        except Exception as e:
            print(f"  Error processing {md_file}: {e}")

    return updated_count


def main():
    parser = argparse.ArgumentParser(
        description="Standardize MinerU image names from hash to readable format"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input directory containing MinerU output"
    )
    parser.add_argument(
        "--update-triplets", "-t",
        type=Path,
        help="Path to triplets.jsonl to update"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--skip-structure",
        action="store_true",
        help="Skip updating structure.json files"
    )
    parser.add_argument(
        "--skip-markdown",
        action="store_true",
        help="Skip updating markdown files"
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input directory not found: {args.input}")
        sys.exit(1)

    if args.dry_run:
        print("=== DRY RUN MODE - No changes will be made ===\n")

    # Step 1: Standardize image names
    print("Step 1: Renaming hash-based image files...")
    rename_map = standardize_images(args.input, dry_run=args.dry_run)
    print(f"  Total images renamed: {len(rename_map)}\n")

    if not rename_map:
        print("No hash-based image names found. Nothing to update.")
        return

    # Step 2: Update structure.json
    if not args.skip_structure:
        print("Step 2: Updating structure.json files...")
        structure_count = update_structure_json(
            args.input, rename_map, dry_run=args.dry_run
        )
        print(f"  Updated {structure_count} structure files\n")

    # Step 3: Update markdown files
    if not args.skip_markdown:
        print("Step 3: Updating markdown files...")
        md_count = update_markdown_files(
            args.input, rename_map, dry_run=args.dry_run
        )
        print(f"  Updated {md_count} markdown files\n")

    # Step 4: Update triplets if specified
    if args.update_triplets:
        print("Step 4: Updating triplets.jsonl...")
        triplet_count = update_triplets_jsonl(
            args.update_triplets, rename_map, dry_run=args.dry_run
        )
        print(f"  Updated {triplet_count} triplets\n")

    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Images renamed: {len(rename_map)}")
    if args.dry_run:
        print("\n(Dry run - no actual changes were made)")
    else:
        print("\nAll files have been updated successfully.")

    # Save rename mapping for reference
    if not args.dry_run and rename_map:
        mapping_file = args.input / "image_rename_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(rename_map, f, ensure_ascii=False, indent=2)
        print(f"\nRename mapping saved to: {mapping_file}")


if __name__ == "__main__":
    main()
