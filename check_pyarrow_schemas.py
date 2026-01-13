#!/usr/bin/env python3
"""
Check PyArrow schemas in actual shard files vs field mappings.

This script helps identify:
- Column name mismatches between field mappings and actual PyArrow schemas
- Missing columns in shard files
- Inconsistent schemas across shards
- Which shards have the problematic schema
"""

import sys
from pathlib import Path
from collections import defaultdict
import pyarrow as pa


def get_pyarrow_schema(shard_path: Path):
    """Get the column names from a PyArrow shard file."""
    try:
        reader = pa.RecordBatchFileReader(pa.memory_map(str(shard_path)))
        return set(reader.schema.names)
    except Exception as e:
        return None, str(e)


def check_subdirectory_schemas(dataset_path: Path, subdir_name: str, max_shards: int = 20):
    """Check schemas in a subdirectory."""
    subdir_path = dataset_path / subdir_name
    
    if not subdir_path.exists() or not subdir_path.is_dir():
        return None
    
    shard_files = sorted([f for f in subdir_path.iterdir() if f.suffix == ".wsds"])
    
    if not shard_files:
        return None
    
    # Check up to max_shards (or all if fewer)
    shards_to_check = shard_files[:max_shards]
    
    schemas = {}
    errors = {}
    
    for shard_file in shards_to_check:
        schema = get_pyarrow_schema(shard_file)
        if isinstance(schema, tuple) and schema[0] is None:
            # Error occurred
            errors[shard_file.name] = schema[1]
        else:
            schemas[shard_file.name] = schema
    
    return {
        'total_shards': len(shard_files),
        'checked_shards': len(shards_to_check),
        'schemas': schemas,
        'errors': errors
    }


def analyze_dataset_schemas(dataset_path: str, max_shards_per_subdir: int = 20):
    """Analyze PyArrow schemas across all subdirectories."""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Analyzing PyArrow Schemas: {dataset_path}")
    print(f"{'='*80}\n")
    
    # Get field mappings
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from hume_wsds import WSDataset
        
        try:
            ds = WSDataset(str(dataset_path))
            print(f"Field mappings from WSDataset:")
            for field, (subdir, col) in sorted(ds.fields.items()):
                print(f"  {field} -> {subdir}/{col}")
            print()
        except Exception as e:
            print(f"⚠️  Could not load WSDataset: {e}\n")
            ds = None
    except ImportError:
        print("⚠️  Could not import WSDataset\n")
        ds = None
    
    # Get all subdirectories
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    if not subdirs:
        print(f"WARNING: No subdirectories found in {dataset_path}")
        return
    
    print(f"Checking schemas in {len(subdirs)} subdirectories (sampling up to {max_shards_per_subdir} shards each)...\n")
    
    all_results = {}
    
    for subdir in sorted(subdirs):
        print(f"{'='*80}")
        print(f"Subdirectory: {subdir.name}")
        print(f"{'='*80}\n")
        
        result = check_subdirectory_schemas(dataset_path, subdir.name, max_shards_per_subdir)
        
        if result is None:
            print(f"  ⚠️  No shards found or directory doesn't exist\n")
            continue
        
        all_results[subdir.name] = result
        
        print(f"  Total shards: {result['total_shards']}")
        print(f"  Checked shards: {result['checked_shards']}\n")
        
        if result['errors']:
            print(f"  ❌ ERRORS reading {len(result['errors'])} shards:")
            for shard, error in list(result['errors'].items())[:5]:
                print(f"    {shard}: {error}")
            if len(result['errors']) > 5:
                print(f"    ... and {len(result['errors']) - 5} more errors")
            print()
        
        if not result['schemas']:
            print(f"  ⚠️  No valid schemas found\n")
            continue
        
        # Analyze schema consistency
        all_columns = set()
        for schema in result['schemas'].values():
            all_columns.update(schema)
        
        print(f"  Columns found across all checked shards: {len(all_columns)}")
        for col in sorted(all_columns):
            print(f"    - {col}")
        print()
        
        # Check for inconsistent schemas
        schema_variants = defaultdict(list)
        for shard_name, schema in result['schemas'].items():
            schema_key = tuple(sorted(schema))
            schema_variants[schema_key].append(shard_name)
        
        if len(schema_variants) > 1:
            print(f"  ⚠️  WARNING: Found {len(schema_variants)} different schema variants!")
            for i, (schema_cols, shards) in enumerate(schema_variants.items(), 1):
                print(f"\n  Variant {i} ({len(shards)} shards):")
                print(f"    Columns: {sorted(schema_cols)}")
                print(f"    Shards: {', '.join(sorted(shards)[:5])}")
                if len(shards) > 5:
                    print(f"            ... and {len(shards) - 5} more")
        else:
            print(f"  ✓ All checked shards have consistent schemas\n")
        
        # Compare with field mappings if available
        if ds is not None:
            print(f"  Field mapping comparison:")
            expected_columns = set()
            for field, (mapped_subdir, mapped_col) in ds.fields.items():
                if mapped_subdir == subdir.name:
                    expected_columns.add(mapped_col)
            
            if expected_columns:
                actual_columns = all_columns
                missing = expected_columns - actual_columns
                extra = actual_columns - expected_columns
                
                if missing:
                    print(f"    ❌ Missing columns (expected but not found):")
                    for col in sorted(missing):
                        print(f"      - {col}")
                
                if extra:
                    print(f"    ⚠️  Extra columns (found but not in mapping):")
                    for col in sorted(extra):
                        print(f"      - {col}")
                
                if not missing and not extra:
                    print(f"    ✓ All expected columns found")
                print()
        
        print()
    
    # Summary
    print(f"{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    issues_found = False
    
    for subdir_name, result in all_results.items():
        if result['errors']:
            issues_found = True
            print(f"❌ {subdir_name}: {len(result['errors'])} shards had read errors")
        
        if result['schemas']:
            schema_variants = defaultdict(list)
            for shard_name, schema in result['schemas'].items():
                schema_key = tuple(sorted(schema))
                schema_variants[schema_key].append(shard_name)
            
            if len(schema_variants) > 1:
                issues_found = True
                print(f"⚠️  {subdir_name}: {len(schema_variants)} different schema variants found")
    
    if not issues_found:
        print("✓ No schema inconsistencies found in checked shards")
        print("  (Note: Only sampled shards were checked. Run with higher max_shards to check all)")
    else:
        print("\n⚠️  Issues found! Some shards may have inconsistent schemas.")
        print("  This could cause 'Field does not exist' errors when accessing columns.")
    
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check PyArrow schemas in WSDS shard files"
    )
    parser.add_argument(
        "dataset_path",
        help="Path to the dataset directory (e.g., /path/to/source)"
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=20,
        help="Maximum number of shards to check per subdirectory (default: 20)"
    )
    
    args = parser.parse_args()
    
    analyze_dataset_schemas(args.dataset_path, args.max_shards)

