import pandas as pd
from utils import *
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    EarlyStoppingCallback
) 
import json
import os


def _parse_override_value(raw_value: str):
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        lowered = raw_value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "null":
            return None
        return raw_value


def _apply_config_override(config: dict, override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Invalid override format: {override}. Expected key=value")

    key_path, raw_value = override.split("=", 1)
    key_parts = [part.strip() for part in key_path.split(".") if part.strip()]
    if not key_parts:
        raise ValueError(f"Invalid override key: {key_path}")

    current = config
    for part in key_parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]

    current[key_parts[-1]] = _parse_override_value(raw_value)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="config file")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Đường dẫn đến file cấu hình (.json)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config bằng key=value, hỗ trợ dot notation như training.metric_for_best_model=\"aoste_micro_f1\"",
    )
    return parser.parse_args()


def get_loaded_instruction_handler(handler_type, instruction_type):
    handler = None
    match handler_type:
        case "InstructionHandler":
            handler = InstructionHandler()
        case "InstructionSpanHandler":
            handler = InstructionSpanHandler()
        case "InstructionSegmentHandler":
            handler = InstructionSegmentHandler()
        case _ :
            raise ValueError(f"Unknown handler type: {handler_type}")
    
    match instruction_type:
        case "type_0":
            handler.load_instruction_0()
        case "type_1":
            handler.load_instruction_1()
        case "type_2":
            handler.load_instruction_2()
        case "type_2_modified" :
            handler.load_instruction_2_modified()
        case "type_3":
            if handler_type not in ["InstructionHandler", "InstructionSegmentHandler"]:
                raise ValueError("Type 3 instruction can only be used with InstructionHandler or InstructionSegmentHandler")
            handler.load_instruction_3()
        case _:
            raise ValueError(f"Unknown instruction type: {instruction_type}")
    
    return handler


def _sanitize_metric_name(metric_name: str) -> str:
    return metric_name.replace("/", "_").replace(" ", "_")


def _get_eval_metric_keys(metrics: dict) -> list[str]:
    excluded = {
        "eval_runtime",
        "eval_samples_per_second",
        "eval_steps_per_second",
        "eval_jit_compilation_time",
        "epoch"
    }
    keys = []
    for k, v in metrics.items():
        if not k.startswith("eval_"):
            continue
        if k in excluded:
            continue
        if isinstance(v, (int, float)):
            keys.append(k)
    return keys


class BestPerMetricSaveCallback(TrainerCallback):
    def __init__(self, tokenizer, save_root, tracked_metrics=None):
        self.tokenizer = tokenizer
        self.save_root = save_root
        self.best_scores = {}
        self.best_model_paths = {}
        self.tracked_metrics = set(tracked_metrics) if tracked_metrics else None

    @staticmethod
    def _is_better(metric_name: str, new_score: float, old_score: float) -> bool:
        if metric_name.endswith("loss"):
            return new_score < old_score
        return new_score > old_score

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if metrics is None or model is None:
            return control

        os.makedirs(self.save_root, exist_ok=True)
        metric_keys = _get_eval_metric_keys(metrics)
        if self.tracked_metrics is not None:
            metric_keys = [k for k in metric_keys if k in self.tracked_metrics]

        for metric_name in metric_keys:
            score = metrics[metric_name]
            prev = self.best_scores.get(metric_name)
            should_save = prev is None or self._is_better(metric_name, score, prev)
            if not should_save:
                continue

            self.best_scores[metric_name] = score
            metric_dir = os.path.join(self.save_root, _sanitize_metric_name(metric_name))
            os.makedirs(metric_dir, exist_ok=True)

            model.save_pretrained(metric_dir)
            self.tokenizer.save_pretrained(metric_dir)

            with open(os.path.join(metric_dir, "best_score.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "metric": metric_name,
                        "best_score": score,
                        "global_step": state.global_step,
                        "epoch": state.epoch,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            self.best_model_paths[metric_name] = metric_dir

        return control


def _decode_predict_output(result, tokenizer):
    preds = result.predictions
    labels = result.label_ids
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    return decoded_preds, decoded_labels


def _run_inference_and_save(
    trainer,
    tokenizer,
    tokenized_test_dataset,
    raw_test_dataset,
    result_dir,
    prediction_name,
    metrics_name,
):
    result = trainer.predict(tokenized_test_dataset)
    decoded_preds, decoded_labels = _decode_predict_output(result, tokenizer)

    data_dict = {
        "text": [raw_test_dataset[i]["input"].split("input: ")[-1].split("\noutput:")[0] for i in range(len(raw_test_dataset))],
        "predict": decoded_preds,
        "label": decoded_labels,
    }
    result_df = pd.DataFrame(data_dict)
    result_df.to_csv(os.path.join(result_dir, prediction_name), index=False)

    with open(os.path.join(result_dir, metrics_name), "w", encoding="utf-8") as f:
        json.dump(result.metrics, f, ensure_ascii=False, indent=2)


def main(config):
    # create model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained"], use_fast = True) 
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model"]["pretrained"])

    new_tokens = ["<IA>", "<IO>", "##", "$"]
    num_added = tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # prepare data
    data_dir = config["data"]["data_dir"]
    allow_punctuation = config["data"]["allow_punctuation"]

    instruction_handler = get_loaded_instruction_handler(config["data"]["handler_type"], config["data"]["instruction_type"])

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    train_df = create_data_with_task(train_df)
    train_dataset = get_dataset_for_training(train_df, instruction_handler, allow_punctuation )

    dev_df = pd.read_csv(os.path.join(data_dir, "dev.csv"))
    dev_df = create_data_with_task(dev_df)
    dev_dataset = get_dataset_for_training(dev_df, instruction_handler, allow_punctuation )

    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    test_df = create_data_with_task(test_df)
    test_dataset = get_dataset_for_training(test_df, instruction_handler, allow_punctuation )

    print(f"Train: {len(train_dataset)}")
    print(f"Dev: {len(dev_dataset)}")
    print(f"Test: {len(test_dataset)}")

    def preprocess_function(examples):
        model_input = tokenizer(
            text = examples['input'],
            text_target = examples['output'],
            truncation = True,
            max_length = config["model"]["max_length"]
        )

        return model_input

    tokenized_train_dataset = train_dataset.map(preprocess_function, batched = True, remove_columns=["input", "output"])
    tokenized_dev_dataset = dev_dataset.map(preprocess_function, batched = True, remove_columns=["input", "output"])
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched = True, remove_columns=["input", "output"])

    # prepare for training
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    metric_types = config["metric_type"] if isinstance(config["metric_type"], list) else [config["metric_type"]]
    compute_metrics = get_metric_fn(tokenizer=tokenizer, metric_type=metric_types)
    
    training_cfg = dict(config["training"])

    metric_for_best_model = training_cfg.get("metric_for_best_model")
    if metric_for_best_model is None:
        raise ValueError("training.metric_for_best_model is required")

    metric_base = str(metric_for_best_model)
    if metric_base.startswith("eval_"):
        metric_base = metric_base[5:]

    tracked_metrics_by_type = {}
    for i, metric_type in enumerate(metric_types):
        if i == 0:
            tracked_metrics_by_type[metric_type] = f"eval_{metric_base}"
        else:
            tracked_metrics_by_type[metric_type] = f"eval_{metric_type}_{metric_base}"

    # Keep only custom best-per-metric checkpoints to save disk space.
    training_cfg["save_strategy"] = "no"
    training_cfg["load_best_model_at_end"] = False

    training_args = Seq2SeqTrainingArguments(
        disable_tqdm=False, 
        **training_cfg
    )

    result_dir = config.get("result_dir", "./result")
    os.makedirs(result_dir, exist_ok=True)

    best_models_root = os.path.join(result_dir, "best_models")
    best_callback = BestPerMetricSaveCallback(
        tokenizer=tokenizer,
        save_root=best_models_root,
        tracked_metrics=list(tracked_metrics_by_type.values()),
    )

    # assemble callbacks
    callbacks = []
    if best_callback is not None:
        callbacks.append(best_callback)

    early_patience = training_cfg.get("early_stopping_patience", 10)
    early_threshold = training_cfg.get("early_stopping_threshold", 1e-4)
    callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_patience, early_stopping_threshold=early_threshold))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_dev_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # train
    trainer.train()
    
    # Inference for each best model by metric type
    best_predictions_dir = os.path.join(result_dir, "predictions")
    os.makedirs(best_predictions_dir, exist_ok=True)

    best_model_summary = {}
    for metric_type, metric_name in tracked_metrics_by_type.items():
        model_dir = best_callback.best_model_paths.get(metric_name)
        if model_dir is None:
            continue

        metric_slug = _sanitize_metric_name(metric_type)

        metric_model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        trainer.model = metric_model.to(trainer.args.device)

        pred_file = f"predict_{metric_slug}.csv"
        metric_file = f"metrics_{metric_slug}.json"
        _run_inference_and_save(
            trainer=trainer,
            tokenizer=tokenizer,
            tokenized_test_dataset=tokenized_test_dataset,
            raw_test_dataset=test_dataset,
            result_dir=best_predictions_dir,
            prediction_name=pred_file,
            metrics_name=metric_file,
        )

        best_model_summary[metric_type] = {
            "tracked_metric": metric_name,
            "model_dir": model_dir,
            "best_score": best_callback.best_scores[metric_name],
            "prediction_file": os.path.join(best_predictions_dir, pred_file),
            "metrics_file": os.path.join(best_predictions_dir, metric_file),
        }

    with open(os.path.join(result_dir, "best_model_summary.json"), "w", encoding="utf-8") as f:
        json.dump(best_model_summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    for override in args.set:
        _apply_config_override(config, override)

    seed = config.get("seed", None)
    set_seed(seed)
    main(config)


    
