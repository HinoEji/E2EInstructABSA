import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = {
    "max_length": 512,
    "handler_type": "InstructionSpanHandler",
    "instruction_type": "type_2",
    "allow_punctuation": True,
    "metric_types": ["strict", "lcs", "loose"],
    "eval_batch_size": 16,
    "generation_max_length": 256,
    "generation_num_beams": 2,
    "fp16": False,
}

REQUIRED_DATASET_COLUMNS = {
    "text",
    "aspects",
    "aspect_sentiment_pairs",
    "aspect_opinion_pairs",
    "triplets",
}


def resolve_dataset_path(dataset_path: Path) -> Path:
    """Chuẩn hóa đường dẫn CSV hoặc thư mục chứa test.csv."""
    dataset_path = dataset_path.expanduser()
    if dataset_path.is_dir():
        dataset_path = dataset_path / "test.csv"

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if dataset_path.suffix.lower() != ".csv":
        raise ValueError(f"Dataset must be a CSV file: {dataset_path}")

    return dataset_path


def build_runtime_config(config_path: Path | None) -> dict[str, Any]:
    """Đọc các thiết lập suy luận tương thích từ config huấn luyện."""
    runtime_config = dict(DEFAULT_CONFIG)
    if config_path is None:
        return runtime_config

    with config_path.expanduser().open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})

    runtime_config.update(
        {
            "max_length": model_config.get("max_length", runtime_config["max_length"]),
            "handler_type": data_config.get(
                "handler_type", runtime_config["handler_type"]
            ),
            "instruction_type": data_config.get(
                "instruction_type", runtime_config["instruction_type"]
            ),
            "allow_punctuation": data_config.get(
                "allow_punctuation", runtime_config["allow_punctuation"]
            ),
            "metric_types": config.get("metric_type", runtime_config["metric_types"]),
            "eval_batch_size": training_config.get(
                "per_device_eval_batch_size", runtime_config["eval_batch_size"]
            ),
            "generation_max_length": training_config.get(
                "generation_max_length", runtime_config["generation_max_length"]
            ),
            "generation_num_beams": training_config.get(
                "generation_num_beams", runtime_config["generation_num_beams"]
            ),
            "fp16": training_config.get("fp16", runtime_config["fp16"]),
        }
    )

    if isinstance(runtime_config["metric_types"], str):
        runtime_config["metric_types"] = [runtime_config["metric_types"]]

    return runtime_config


def parse_args() -> argparse.Namespace:
    """Phân tích tham số dòng lệnh cho dự đoán và đánh giá."""
    parser = argparse.ArgumentParser(
        description="Predict and evaluate an ABSA CSV dataset with a trained model."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="CSV file or directory containing test.csv.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Training config used to preserve instruction and generation settings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prediction_result"),
    )
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--generation-max-length", type=int)
    parser.add_argument("--generation-num-beams", type=int)
    parser.add_argument(
        "--metric-type",
        nargs="+",
        choices=["strict", "lcs", "loose"],
    )
    return parser.parse_args()


def apply_cli_overrides(
    runtime_config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Áp dụng các tùy chọn suy luận được truyền trực tiếp từ CLI."""
    overrides = {
        "eval_batch_size": args.eval_batch_size,
        "generation_max_length": args.generation_max_length,
        "generation_num_beams": args.generation_num_beams,
        "metric_types": args.metric_type,
    }
    for key, value in overrides.items():
        if value is not None:
            runtime_config[key] = value
    return runtime_config


def validate_dataset_columns(columns: list[str]) -> None:
    """Kiểm tra dataset có đủ nhãn cần thiết để đánh giá."""
    missing_columns = REQUIRED_DATASET_COLUMNS.difference(columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")


def run_prediction(
    model_path: Path,
    dataset_path: Path,
    output_dir: Path,
    runtime_config: dict[str, Any],
) -> None:
    """Chạy sinh kết quả, tính metric và ghi kết quả ra đĩa."""
    import numpy as np
    import pandas as pd
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    from train import get_loaded_instruction_handler
    from utils import create_data_with_task, get_dataset_for_training, get_metric_fn

    model_path = model_path.expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    source_df = pd.read_csv(dataset_path)
    validate_dataset_columns(source_df.columns.tolist())

    task_df = create_data_with_task(source_df)
    instruction_handler = get_loaded_instruction_handler(
        runtime_config["handler_type"],
        runtime_config["instruction_type"],
    )
    raw_dataset = get_dataset_for_training(
        task_df,
        instruction_handler,
        runtime_config["allow_punctuation"],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    def preprocess_function(examples):
        """Mã hóa cả instruction đầu vào và nhãn chuẩn."""
        return tokenizer(
            text=examples["input"],
            text_target=examples["output"],
            truncation=True,
            max_length=runtime_config["max_length"],
        )

    tokenized_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["input", "output"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    use_fp16 = bool(runtime_config["fp16"] and torch.cuda.is_available())
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "trainer_output"),
        per_device_eval_batch_size=runtime_config["eval_batch_size"],
        predict_with_generate=True,
        generation_max_length=runtime_config["generation_max_length"],
        generation_num_beams=runtime_config["generation_num_beams"],
        fp16=use_fp16,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=get_metric_fn(
            tokenizer=tokenizer,
            metric_type=runtime_config["metric_types"],
        ),
    )

    result = trainer.predict(tokenized_dataset)
    predictions = np.where(
        result.predictions != -100,
        result.predictions,
        tokenizer.pad_token_id,
    )
    labels = np.where(
        result.label_ids != -100,
        result.label_ids,
        tokenizer.pad_token_id,
    )
    decoded_predictions = [
        value.strip()
        for value in tokenizer.batch_decode(predictions, skip_special_tokens=True)
    ]
    decoded_labels = [
        value.strip()
        for value in tokenizer.batch_decode(labels, skip_special_tokens=True)
    ]

    prediction_df = pd.DataFrame(
        {
            "task": task_df["Task"].tolist(),
            "input": task_df["Input"].tolist(),
            "predict": decoded_predictions,
            "label": decoded_labels,
        }
    )
    prediction_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"
    prediction_df.to_csv(prediction_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(result.metrics, metrics_file, ensure_ascii=False, indent=2)

    print(f"Predictions: {prediction_path}")
    print(f"Metrics: {metrics_path}")


def main() -> None:
    """Điều phối toàn bộ luồng dự đoán từ tham số CLI."""
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset_path)
    runtime_config = apply_cli_overrides(
        build_runtime_config(args.config),
        args,
    )
    run_prediction(
        model_path=args.model_path,
        dataset_path=dataset_path,
        output_dir=args.output_dir.expanduser(),
        runtime_config=runtime_config,
    )


if __name__ == "__main__":
    main()
