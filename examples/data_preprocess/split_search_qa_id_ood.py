"""Create deterministic combined Search-QA validation and test parquet files.

ID/OOD is kept as a source label for evaluation metrics; it is not materialized
as separate parquet files.
"""

import argparse
import os

import pandas as pd


def normalized_source(value):
    source = str(value).lower()
    if source.startswith("searchr1_"):
        source = source[len("searchr1_") :]
    return source


def balanced_sample(dataframe, max_samples, seed):
    """Sample up to max_samples while balancing available data sources."""
    if len(dataframe) <= max_samples:
        return dataframe.sample(frac=1, random_state=seed).reset_index(drop=True)

    groups = {
        source: group
        for source, group in dataframe.groupby(dataframe["data_source"].map(normalized_source))
    }
    sources = sorted(groups)
    allocation = {source: min(len(groups[source]), max_samples // len(sources)) for source in sources}
    remaining = max_samples - sum(allocation.values())

    while remaining > 0:
        eligible = [source for source in sources if allocation[source] < len(groups[source])]
        if not eligible:
            break
        per_source = max(1, remaining // len(eligible))
        for source in eligible:
            add = min(per_source, remaining, len(groups[source]) - allocation[source])
            allocation[source] += add
            remaining -= add
            if remaining == 0:
                break

    sampled = [
        groups[source].sample(n=count, random_state=seed + index)
        for index, (source, count) in enumerate(allocation.items())
        if count > 0
    ]
    return pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Create combined Search-QA val.parquet and test.parquet files."
    )
    parser.add_argument("--input", required=True, help="Processed Search-QA test.parquet.")
    parser.add_argument("--output_dir", required=True)
    # Kept for compatibility with scripts/prepare_search_qa.sh.  The two
    # source groups together form one validation parquet.
    parser.add_argument("--val_samples_per_group", type=int, default=1000)
    parser.add_argument("--val_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = os.path.abspath(os.path.expanduser(args.input))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    dataframe = pd.read_parquet(input_path)
    test_combined = dataframe.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    val_size = args.val_samples if args.val_samples is not None else 2 * args.val_samples_per_group
    val_combined = balanced_sample(dataframe, val_size, args.seed)

    outputs = {
        "test.parquet": test_combined,
        "val.parquet": val_combined,
    }
    for filename, split in outputs.items():
        path = os.path.join(output_dir, filename)
        split.to_parquet(path, index=False)
        counts = split["data_source"].value_counts().to_dict()
        print(f"Saved {filename}: rows={len(split)}, sources={counts}")


if __name__ == "__main__":
    main()
