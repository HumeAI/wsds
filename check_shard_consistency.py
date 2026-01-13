#!/usr/bin/env python3
"""
Diagnostic script to check shard consistency across subdirectories in a WSDS dataset.

This script helps identify:
- Which shards exist in which subdirectories
- Missing shards (shards that exist in some subdirs but not others)
- Extra shards (shards that only exist in one subdirectory)
- Field mappings (especially __key__)
- Potential issues that could cause the "Field does not exist" error
"""

import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set


def get_shards_in_subdir(subdir: Path) -> Set[str]:
    """Get all .wsds shard names (without extension) in a subdirectory."""
    if not subdir.is_dir():
        return set()
    return {f.stem for f in subdir.iterdir() if f.suffix == ".wsds"}


def get_all_subdirs(dataset_path: Path) -> Dict[str, Path]:
    """Get all subdirectories in the dataset."""
    subdirs = {}
    for item in dataset_path.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            subdirs[item.name] = item
    return subdirs


def analyze_dataset(dataset_path: str):
    """Analyze a WSDS dataset for shard consistency issues."""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Analyzing dataset: {dataset_path}")
    print(f"{'='*80}\n")
    
    # Get all subdirectories
    subdirs = get_all_subdirs(dataset_path)
    if not subdirs:
        print(f"WARNING: No subdirectories found in {dataset_path}")
        return
    
    print(f"Found {len(subdirs)} subdirectories:")
    for name in sorted(subdirs.keys()):
        print(f"  - {name}")
    print()
    
    # Get shards for each subdirectory
    subdir_shards: Dict[str, Set[str]] = {}
    for name, path in subdirs.items():
        shards = get_shards_in_subdir(path)
        subdir_shards[name] = shards
        print(f"{name}: {len(shards)} shards")
    
    print()
    
    # Find all unique shard names across all subdirectories
    all_shards = set()
    for shards in subdir_shards.values():
        all_shards.update(shards)
    
    print(f"Total unique shard names across all subdirectories: {len(all_shards)}")
    print()
    
    # Check for shards that exist in some subdirs but not others
    print(f"{'='*80}")
    print("SHARD CONSISTENCY ANALYSIS")
    print(f"{'='*80}\n")
    
    issues_found = False
    
    # For each shard, check which subdirectories have it
    shard_locations: Dict[str, Set[str]] = defaultdict(set)
    for subdir_name, shards in subdir_shards.items():
        for shard in shards:
            shard_locations[shard].add(subdir_name)
    
    # Find shards that don't exist in all subdirectories
    incomplete_shards = []
    for shard, locations in shard_locations.items():
        if len(locations) < len(subdirs):
            incomplete_shards.append((shard, locations))
    
    if incomplete_shards:
        issues_found = True
        print(f"⚠️  FOUND {len(incomplete_shards)} SHARDS THAT DON'T EXIST IN ALL SUBDIRECTORIES:\n")
        for shard, locations in sorted(incomplete_shards):
            missing_in = set(subdirs.keys()) - locations
            print(f"  Shard: {shard}")
            print(f"    ✓ Exists in: {', '.join(sorted(locations))}")
            print(f"    ✗ Missing in: {', '.join(sorted(missing_in))}")
            print()
    else:
        print("✓ All shards exist in all subdirectories\n")
    
    # Find shards that only exist in one subdirectory (extra shards)
    extra_shards = []
    for shard, locations in shard_locations.items():
        if len(locations) == 1:
            extra_shards.append((shard, list(locations)[0]))
    
    if extra_shards:
        issues_found = True
        print(f"⚠️  FOUND {len(extra_shards)} SHARDS THAT ONLY EXIST IN ONE SUBDIRECTORY:\n")
        # Group by subdirectory
        by_subdir = defaultdict(list)
        for shard, subdir in extra_shards:
            by_subdir[subdir].append(shard)
        
        for subdir, shards in sorted(by_subdir.items()):
            print(f"  {subdir} has {len(shards)} unique shards:")
            for shard in sorted(shards)[:10]:  # Show first 10
                print(f"    - {shard}")
            if len(shards) > 10:
                print(f"    ... and {len(shards) - 10} more")
            print()
    else:
        print("✓ No extra shards found (all shards exist in multiple subdirectories)\n")
    
    # Check field mappings (if we can import the dataset)
    print(f"{'='*80}")
    print("FIELD MAPPING ANALYSIS")
    print(f"{'='*80}\n")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from hume_wsds import WSDataset
        
        try:
            ds = WSDataset(str(dataset_path))
            
            print(f"Total fields: {len(ds.fields)}\n")
            
            # Show __key__ mapping
            if "__key__" in ds.fields:
                subdir, column = ds.fields["__key__"]
                print(f"__key__ field mapping:")
                print(f"  Subdirectory: {subdir}")
                print(f"  Column name: {column}")
                
                # Check if this subdirectory has all shards
                key_shards = get_shards_in_subdir(dataset_path / subdir)
                if key_shards != all_shards:
                    issues_found = True
                    missing = all_shards - key_shards
                    extra = key_shards - all_shards
                    print(f"  ⚠️  WARNING: __key__ subdirectory doesn't have all shards!")
                    if missing:
                        print(f"     Missing shards: {len(missing)} (showing first 5)")
                        for s in sorted(missing)[:5]:
                            print(f"       - {s}")
                    if extra:
                        print(f"     Extra shards: {len(extra)} (showing first 5)")
                        for s in sorted(extra)[:5]:
                            print(f"       - {s}")
                else:
                    print(f"  ✓ __key__ subdirectory has all {len(key_shards)} shards")
                print()
            
            # Show audio field mappings
            audio_fields = [f for f in ds.fields.keys() if f in ["mp3", "flac", "wav", "m4a", "ogg", "wma", "opus", "audio"]]
            if audio_fields:
                print(f"Audio fields found: {', '.join(sorted(audio_fields))}")
                for field in sorted(audio_fields):
                    subdir, column = ds.fields[field]
                    print(f"  {field}: {subdir}/{column}")
                print()
            
            # Show which subdirectories have which fields
            print("Fields by subdirectory:")
            by_subdir = defaultdict(list)
            for field, (subdir, col) in ds.fields.items():
                by_subdir[subdir].append(field)
            
            for subdir in sorted(by_subdir.keys()):
                fields = by_subdir[subdir]
                print(f"  {subdir}: {len(fields)} fields")
                if len(fields) <= 10:
                    for f in sorted(fields):
                        print(f"    - {f}")
                else:
                    for f in sorted(fields)[:10]:
                        print(f"    - {f}")
                    print(f"    ... and {len(fields) - 10} more")
            print()
            
        except Exception as e:
            print(f"⚠️  Could not load dataset (this is OK if index doesn't exist yet): {e}\n")
    
    except ImportError:
        print("⚠️  Could not import WSDataset (make sure you're in the wsds directory)\n")
    
    # Summary
    print(f"{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    if issues_found:
        print("❌ ISSUES FOUND:")
        print("   The dataset has shard inconsistencies that could cause errors.")
        print("   When processing shards that don't exist in all subdirectories,")
        print("   accessing fields from missing subdirectories will fail.\n")
        print("   RECOMMENDATION:")
        print("   - Ensure all shards exist in all subdirectories, OR")
        print("   - Only process shards that exist in the subdirectory you need\n")
    else:
        print("✓ No shard consistency issues found!")
        print("   All shards exist in all subdirectories.\n")
    
    # Show shard count per subdirectory
    print("Shard counts per subdirectory:")
    for subdir_name in sorted(subdir_shards.keys()):
        count = len(subdir_shards[subdir_name])
        print(f"  {subdir_name}: {count} shards")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_shard_consistency.py <dataset_path>")
        print("\nExample:")
        print("  python check_shard_consistency.py /path/to/dataset/source")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    analyze_dataset(dataset_path)

