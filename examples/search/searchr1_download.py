"""Download and assemble the Wikipedia index used by Search-QA."""

import argparse
import gzip
import os
import shutil
import tarfile

from huggingface_hub import hf_hub_download


def unwrap_tar_corpus(corpus_path):
    """Replace a tar-wrapped corpus with its contained JSONL file."""
    if not tarfile.is_tarfile(corpus_path):
        return False

    partial_path = f"{corpus_path}.jsonl.partial"
    if os.path.exists(partial_path):
        os.remove(partial_path)

    with tarfile.open(corpus_path, mode="r:") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.lower().endswith(".jsonl")
        ]
        if len(members) != 1:
            raise RuntimeError(
                f"Expected exactly one JSONL file in {corpus_path}, found {len(members)}"
            )
        source = archive.extractfile(members[0])
        if source is None:
            raise RuntimeError(f"Could not open {members[0].name} inside {corpus_path}")
        with source, open(partial_path, "wb") as output:
            shutil.copyfileobj(source, output, length=16 * 1024 * 1024)

    if os.path.getsize(partial_path) != members[0].size:
        raise RuntimeError(
            f"Extracted corpus size mismatch: expected {members[0].size}, "
            f"got {os.path.getsize(partial_path)}"
        )
    os.replace(partial_path, corpus_path)
    print(f"Unwrapped tar member {members[0].name}: {corpus_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download the Search-R1 E5 index and Wikipedia corpus.")
    parser.add_argument("--local_dir", required=True, help="Destination directory.")
    parser.add_argument("--index_repo_id", default="PeterJinGo/wiki-18-e5-index")
    parser.add_argument("--corpus_repo_id", default="PeterJinGo/wiki-18-corpus")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not access Hugging Face; assemble already-downloaded part_aa, part_ab, and wiki-18.jsonl.gz.",
    )
    parser.add_argument(
        "--unwrap_only",
        action="store_true",
        help="Only unwrap an already decompressed tar-wrapped wiki-18.jsonl file.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild assembled files if they already exist.")
    args = parser.parse_args()

    local_dir = os.path.abspath(os.path.expanduser(args.local_dir))
    os.makedirs(local_dir, exist_ok=True)
    if args.unwrap_only:
        corpus_path = os.path.join(local_dir, "wiki-18.jsonl")
        if not os.path.isfile(corpus_path):
            raise FileNotFoundError(f"Missing decompressed corpus: {corpus_path}")
        if not unwrap_tar_corpus(corpus_path):
            print(f"Corpus is already plain JSONL: {corpus_path}")
        return

    parts = []
    for filename in ("part_aa", "part_ab"):
        if args.offline:
            part_path = os.path.join(local_dir, filename)
            if not os.path.isfile(part_path):
                raise FileNotFoundError(f"Missing offline index part: {part_path}")
            parts.append(part_path)
        else:
            parts.append(
                hf_hub_download(
                    repo_id=args.index_repo_id,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=local_dir,
                )
            )

    index_path = os.path.join(local_dir, "e5_Flat.index")
    if args.force or not os.path.exists(index_path):
        with open(index_path, "wb") as output:
            for part_path in parts:
                with open(part_path, "rb") as part:
                    shutil.copyfileobj(part, output)
        print(f"Assembled index: {index_path}")
    else:
        print(f"Index already exists: {index_path}")

    if args.offline:
        corpus_gz = os.path.join(local_dir, "wiki-18.jsonl.gz")
        if not os.path.isfile(corpus_gz):
            raise FileNotFoundError(f"Missing offline corpus archive: {corpus_gz}")
    else:
        corpus_gz = hf_hub_download(
            repo_id=args.corpus_repo_id,
            filename="wiki-18.jsonl.gz",
            repo_type="dataset",
            local_dir=local_dir,
        )
    corpus_path = os.path.join(local_dir, "wiki-18.jsonl")
    if args.force or not os.path.exists(corpus_path):
        with gzip.open(corpus_gz, "rb") as source, open(corpus_path, "wb") as output:
            shutil.copyfileobj(source, output)
        print(f"Extracted corpus: {corpus_path}")
    else:
        print(f"Corpus already exists: {corpus_path}")
    unwrap_tar_corpus(corpus_path)

    print("Search index preparation complete.")


if __name__ == "__main__":
    main()
