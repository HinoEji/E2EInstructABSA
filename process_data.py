import argparse
import ast
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from utils.data_processing import format_data


def parse_sentiments(aspect_sentiment_pairs: Any) -> set:
    if isinstance(aspect_sentiment_pairs, str):
        pairs = ast.literal_eval(aspect_sentiment_pairs)
    else:
        pairs = aspect_sentiment_pairs

    return {sentiment for _, sentiment in pairs}


def collect_example_buckets(raw_df: pd.DataFrame) -> Dict[str, List[int]]:
    buckets = {
        "only_pos": [],
        "only_neg": [],
        "only_neu": [],
        "joint": [],
    }

    for idx in range(len(raw_df)):
        sentiments = parse_sentiments(raw_df.iloc[idx]["aspect_sentiment_pairs"])
        if not sentiments:
            continue

        if len(sentiments) == 3:
            buckets["joint"].append(idx)
        elif len(sentiments) == 1:
            if "POS" in sentiments:
                buckets["only_pos"].append(idx)
            elif "NEG" in sentiments:
                buckets["only_neg"].append(idx)
            elif "NEU" in sentiments:
                buckets["only_neu"].append(idx)

    return buckets


def safe_sample(indices: List[int], size: int, rng: random.Random) -> List[int]:
    if not indices:
        return []
    k = min(size, len(indices))
    return rng.sample(indices, k)


def split_dataframe(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs((train_ratio + dev_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train_ratio + dev_ratio + test_ratio must equal 1.0")

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)

    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)

    train_df = shuffled.iloc[:train_end].reset_index(drop=True)
    dev_df = shuffled.iloc[train_end:dev_end].reset_index(drop=True)
    test_df = shuffled.iloc[dev_end:].reset_index(drop=True)

    return train_df, dev_df, test_df


def process_dataset(
    source_csv: Path,
    output_dir: Path,
    example_size: int,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    print(f"\n=== Processing: {source_csv} -> {output_dir} ===")
    output_dir.mkdir(parents=True, exist_ok=True)
    example_dir = output_dir / "example"
    example_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(source_csv)
    formatted_df = format_data(data)
    raw_path = output_dir / "raw.csv"
    formatted_df.to_csv(raw_path, index=False)
    print(f"Saved raw file: {raw_path}")

    raw_df = formatted_df.copy().reset_index(drop=True)
    buckets = collect_example_buckets(raw_df)

    rng = random.Random(seed)
    pos_idx = safe_sample(buckets["only_pos"], example_size, rng)
    neg_idx = safe_sample(buckets["only_neg"], example_size, rng)
    neu_idx = safe_sample(buckets["only_neu"], example_size, rng)
    joint_idx = safe_sample(buckets["joint"], example_size, rng)

    print(
        "Example bucket sizes "
        f"(available -> selected): "
        f"POS {len(buckets['only_pos'])}->{len(pos_idx)}, "
        f"NEG {len(buckets['only_neg'])}->{len(neg_idx)}, "
        f"NEU {len(buckets['only_neu'])}->{len(neu_idx)}, "
        f"JOINT {len(buckets['joint'])}->{len(joint_idx)}"
    )

    pos_example = raw_df.iloc[pos_idx].copy().reset_index(drop=True) if pos_idx else raw_df.iloc[[]].copy()
    neg_example = raw_df.iloc[neg_idx].copy().reset_index(drop=True) if neg_idx else raw_df.iloc[[]].copy()
    neu_example = raw_df.iloc[neu_idx].copy().reset_index(drop=True) if neu_idx else raw_df.iloc[[]].copy()
    joint_example = raw_df.iloc[joint_idx].copy().reset_index(drop=True) if joint_idx else raw_df.iloc[[]].copy()

    pos_example.to_csv(example_dir / "example_pos.csv", index=False)
    neg_example.to_csv(example_dir / "example_neg.csv", index=False)
    neu_example.to_csv(example_dir / "example_neu.csv", index=False)
    joint_example.to_csv(example_dir / "example_joint.csv", index=False)

    selected = pos_idx + neg_idx + neu_idx + joint_idx
    remaining_df = raw_df.drop(selected).reset_index(drop=True)

    train_df, dev_df, test_df = split_dataframe(
        remaining_df,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_df.to_csv(output_dir / "train.csv", index=False)
    dev_df.to_csv(output_dir / "dev.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(
        f"Saved split files: train={len(train_df)}, dev={len(dev_df)}, test={len(test_df)}"
    )


def resolve_first_csv(datapath: Path) -> Path:
    if datapath.is_file():
        if datapath.suffix.lower() != ".csv":
            raise ValueError(f"datapath must be a CSV file or directory: {datapath}")
        return datapath

    if not datapath.exists() or not datapath.is_dir():
        raise ValueError(f"datapath does not exist or is not a directory: {datapath}")

    csv_files = sorted(datapath.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in datapath: {datapath}")

    return csv_files[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process ABSA data: format raw labels, create examples, and split train/dev/test."
    )
    parser.add_argument(
        "--datapath",
        type=Path,
        required=True,
        help="CSV file path or folder path. If folder, the first CSV file (sorted by name) will be processed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save processed files. Default is the folder containing the source CSV.",
    )
    parser.add_argument(
        "--example-size",
        type=int,
        default=12,
        help="Number of examples to sample for each sentiment bucket.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_csv = resolve_first_csv(args.datapath)
    output_dir = args.output_dir if args.output_dir is not None else source_csv.parent
    print(f"Using source from datapath: {source_csv}")

    process_dataset(
        source_csv=source_csv,
        output_dir=output_dir,
        example_size=args.example_size,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
