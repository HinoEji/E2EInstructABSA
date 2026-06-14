import numpy as np
import random

__all__ = ["get_metric_fn"]
# standard metrics for absa task, strict


def _longest_common_subsequence_length(a: str, b: str) -> int:
    """Compute LCS length between two strings."""
    if not a or not b:
        return 0

    len_b = len(b)
    prev = [0] * (len_b + 1)

    for i in range(1, len(a) + 1):
        curr = [0] * (len_b + 1)
        a_char = a[i - 1]
        for j in range(1, len_b + 1):
            if a_char == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr

    return prev[len_b]


def _compute_single_metric(decoded_preds, decoded_labels, metric_type: str, threshold: float = 0.8):
    if metric_type == "strict":
        return compute_strict_metrics(decoded_preds, decoded_labels)
    elif metric_type == "loose":
        return compute_loose_metrics(decoded_preds, decoded_labels)
    elif metric_type == "lcs":
        return compute_lcs_metrics(decoded_preds, decoded_labels, threshold=threshold)
    else:
        raise ValueError("Invalid metric type, must be 'strict', 'loose' or 'lcs_triplet'")


def get_metric_fn(tokenizer, metric_type: str | list[str] = "strict", threshold: float = 0.8):
    metric_types = metric_type if isinstance(metric_type, list) else [metric_type]

    def compute_metrics(eval_preds)-> dict:
        preds, labels = eval_preds
    
        # 1. Decode PREDICTIONS (Kết quả mô hình dự đoán)
        # Nếu dùng DataCollator padding, preds có thể bị thừa padding token, skip_special_tokens=True sẽ lo việc đó
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        
        # 2. Decode LABELS (Đáp án thật)
        # Quan trọng: DataCollator đã thay padding bằng -100. 
        # Tokenizer không hiểu -100 là gì, nên phải đổi ngược lại về pad_token_id (số 0)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # 3. Chuẩn hóa nhẹ (strip khoảng trắng thừa)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        
        if len(decoded_preds) > 0:
            idx = random.randrange(len(decoded_preds))

            print("\n--- Mẫu dự đoán ---")
            # print(f"Index: {idx}")
            print("Pred:", decoded_preds[idx])
            print("Gold:", decoded_labels[idx])
            print("--------------------------")

        merged_metrics = {}
        for i, metric_name in enumerate(metric_types):
            current_metrics = _compute_single_metric(
                decoded_preds,
                decoded_labels,
                metric_type=metric_name,
                threshold=threshold,
            )

            # Backward compatibility: keep old metric keys for the first metric type.
            if i == 0:
                merged_metrics.update(current_metrics)
            else:
                for k, v in current_metrics.items():
                    merged_metrics[f"{metric_name}_{k}"] = v

        return merged_metrics
    return compute_metrics

def compute_strict_metrics(generated_outputs, target_outputs):

    def format_generated_output(preds, golds):
        task = {
            "ate": {"preds": [], "golds": []},
            "atsc": {"preds": [], "golds": []},
            "aspe": {"preds": [], "golds": []},
            "aooe": {"preds": [], "golds": []},
            "aope": {"preds": [], "golds": []},
            "aoste": {"preds": [], "golds": []}
        }
        for pred, gold in zip(preds, golds):
            if "[ate]" in gold:
                gold = gold.split("[ate]")[-1]
                pred = pred.split("[ate]")[-1]

                g = gold.split(" ## ")
                p = pred.split(" ## ")
                
                task["ate"]["golds"].append(g)
                task["ate"]["preds"].append(p)

            elif "[atsc]" in gold:
                gold = gold.split("[atsc]")[-1]
                pred = pred.split("[atsc]")[-1]

                g = gold.split(" ## ")
                p = pred.split(" ## ")

                task["atsc"]["golds"].append(g)
                task["atsc"]["preds"].append(p)

            elif "[aspe]" in gold:
                gold = gold.split("[aspe]")[-1]
                pred = pred.split("[aspe]")[-1]

                g = gold.split(" ## ")
                p = pred.split(" ## ")
                g = [x.split(" $ ") for x in g]
                p = [x.split(" $ ") for x in p]


                task["aspe"]["golds"].append(g)
                task["aspe"]["preds"].append(p)

            elif "[aooe]" in gold:
                gold = gold.split("[aooe]")[-1]
                pred = pred.split("[aooe]")[-1]

                g = gold.split(" ## ")
                p = pred.split(" ## ")

                task["aooe"]["golds"].append(g)
                task["aooe"]["preds"].append(p)

            elif "[aope]" in gold:
                gold = gold.split("[aope]")[-1]
                pred = pred.split("[aope]")[-1]

                g = gold.split(" ## ")
                p = pred.split(" ## ")
                g = [x.split(" $ ") for x in g]
                p = [x.split(" $ ") for x in p]

                task["aope"]["golds"].append(g)
                task["aope"]["preds"].append(p)

            elif "[aoste]" in gold:
                gold = gold.split("[aoste]")[-1]
                pred = pred.split("[aoste]")[-1]

                g = gold.split(" ## ")
                p = pred.split(" ## ")
                g = [x.split(" $ ") for x in g]
                p = [x.split(" $ ") for x in p]


                task["aoste"]["golds"].append(g)
                task["aoste"]["preds"].append(p)
        return task

    def compute_metrics(preds, golds, prefix=""):
        """
        Calculate Aspect-Based Sentiment Analysis (ABSA) metrics.
        Works for both single-label and pair/triplet tasks.
        """

        assert len(preds) == len(golds), "The length of predictions and golds must be the same."
        N = len(golds)

        # ===== CASE: NO SAMPLE =====
        if N == 0:
            return {
                f"{prefix}_macro_precision": None,
                f"{prefix}_macro_recall": None,
                f"{prefix}_macro_f1": None,
                f"{prefix}_precision": None,
                f"{prefix}_recall": None,
                f"{prefix}_f1": None,
            }
        # ===========================

        total_tp = total_fp = total_fn = 0
        macro_pre = macro_rec = macro_f1 = 0.0

        for pred, gold in zip(preds, golds):

            pred_norm = [tuple(x) if isinstance(x, list) else x for x in pred]
            gold_norm = [tuple(x) if isinstance(x, list) else x for x in gold]

            pred_set = set(pred_norm)
            gold_set = set(gold_norm)

            # ---- empty–empty sentence ----
            if len(pred_set) == 0 and len(gold_set) == 0:
                macro_pre += 1.0
                macro_rec += 1.0
                macro_f1  += 1.0
                continue

            tp = len(pred_set & gold_set)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)

            pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1  = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0

            macro_pre += pre
            macro_rec += rec
            macro_f1  += f1

            total_tp += tp
            total_fp += fp
            total_fn += fn

        macro_pre /= N
        macro_rec /= N
        macro_f1  /= N

        micro_pre = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1  = (
            2 * micro_pre * micro_rec / (micro_pre + micro_rec)
            if (micro_pre + micro_rec) > 0 else 0.0
        )

        return {
            f"{prefix}_macro_precision": macro_pre,
            f"{prefix}_macro_recall": macro_rec,
            f"{prefix}_macro_f1": macro_f1,
            f"{prefix}_precision": micro_pre,
            f"{prefix}_recall": micro_rec,
            f"{prefix}_f1": micro_f1,
        }
        
    new_format = format_generated_output(generated_outputs, target_outputs)
    metrics = {}
    for task_name, value in new_format.items():
        metric = compute_metrics(**value, prefix=task_name)
        metrics.update(metric)
    return metrics



def compute_loose_metrics(generated_outputs, target_outputs):
    """
    Based on the original metrics https://github.com/kevinscaria/InstructABSA/blob/main/InstructABSA/utils.py
    """
    def format_generated_output(preds, golds):
        task = {
            "ate": {"preds": [], "golds": []},
            "atsc": {"preds": [], "golds": []},
            "aspe": {"preds": [], "golds": []},
            "aooe": {"preds": [], "golds": []},
            "aope": {"preds": [], "golds": []},
            "aoste": {"preds": [], "golds": []}
        }
        for pred, gold in zip(preds, golds):
            if "[ate]" in gold:
                gold = gold.split("[ate]")[-1]
                pred = pred.split("[ate]")[-1]
                
                task["ate"]["golds"].append(gold)
                task["ate"]["preds"].append(pred)

            elif "[atsc]" in gold:
                gold = gold.split("[atsc]")[-1]
                pred = pred.split("[atsc]")[-1]

                task["atsc"]["golds"].append(gold)
                task["atsc"]["preds"].append(pred)

            elif "[aspe]" in gold:
                gold = gold.split("[aspe]")[-1]
                pred = pred.split("[aspe]")[-1]

                task["aspe"]["golds"].append(gold)
                task["aspe"]["preds"].append(pred)

            elif "[aooe]" in gold:
                gold = gold.split("[aooe]")[-1]
                pred = pred.split("[aooe]")[-1]

                task["aooe"]["golds"].append(gold)
                task["aooe"]["preds"].append(pred)

            elif "[aope]" in gold:
                gold = gold.split("[aope]")[-1]
                pred = pred.split("[aope]")[-1]

                task["aope"]["golds"].append(gold)
                task["aope"]["preds"].append(pred)

            elif "[aoste]" in gold:
                gold = gold.split("[aoste]")[-1]
                pred = pred.split("[aoste]")[-1]

                task["aoste"]["golds"].append(gold)
                task["aoste"]["preds"].append(pred)
        return task

    def check_match(pred:str,gold:str, is_triplet:bool=False) -> bool:
        """
        matching logic
        """

        # in case pred or gold is empty
        if pred == "" and gold == "":
            return True
        elif pred == "" or gold == "":
            return pred == gold
        else:
            if not is_triplet:
                if gold in pred or pred in gold:
                    return True
            else:
                return pred in gold
        

    def compute_metrics(preds, golds, prefix):
        """
        preds and golds are lists of strings, each string is in the output format of the task such as:
        sản phẩm ## chất lượng
        sản phẩm $ tốt $ POS ## thiết kế $ đẹp $ POS

        atsc: is calculated as classification task
        """
        total_pred = 0
        total_gt = 0
        tp = 0

        # if prefix == "atsc":
        #     macro_pre = precision_score(golds, preds, average='macro', zero_division=0)
        #     macro_rec = recall_score(golds, preds, average='macro', zero_division=0)
        #     macro_f1 = f1_score(golds, preds, average='macro', zero_division=0)
        #     micro_pre = precision_score(golds, preds, average='micro', zero_division=0)
        #     micro_rec = recall_score(golds, preds, average='micro', zero_division=0)
        #     micro_f1 = f1_score(golds, preds, average='micro', zero_division=0)
        #     return {
        #             f"{prefix}_macro_precision": macro_pre,
        #             f"{prefix}_macro_recall": macro_rec,
        #             f"{prefix}_macro_f1": macro_f1,
        #             f"{prefix}_micro_precision": micro_pre,
        #             f"{prefix}_micro_recall": micro_rec,
        #             f"{prefix}_micro_f1": micro_f1,
        #         }
        
        for gold, pred in zip(golds, preds):
            gold_list = gold.split("##")
            pred_list = pred.split("##")

            total_pred += len(pred_list)
            total_gt += len(gold_list)
            
            # mean not triplet extraction
            if prefix != "aoste":
                for gold_val in gold_list:
                    for pred_val in pred_list:
                        if check_match(pred_val, gold_val, is_triplet=False):
                            tp += 1
                            break
            else:
            # with triplet extraction task
                for gold_val in gold_list:
                    gold_components = gold_val.split("$")
                    if len(gold_components) != 3:
                        continue

                    gold_aspect, gold_opinion, gold_sentiment = gold_components
                    
                    for pred_val in pred_list:
                        pred_components = pred_val.split("$")
                        if len(pred_components) != 3:
                            continue

                        pred_aspect, pred_opinion, pred_sentiment = pred_components

                        if check_match(pred_aspect, gold_aspect, is_triplet=True) and check_match(pred_opinion, gold_opinion, is_triplet=True) and check_match(pred_sentiment, gold_sentiment, is_triplet=True):
                            tp += 1
                            break
        
        p = tp/total_pred if total_pred > 0 else 0
        r = tp/total_gt if total_gt > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        return {
            f"{prefix}_precision": p,
            f"{prefix}_recall": r,
            f"{prefix}_f1": f1,
        }

    new_format = format_generated_output(generated_outputs, target_outputs)
    metrics = {}
    for task_name, value in new_format.items():
        metric = compute_metrics(**value, prefix=task_name)
        metrics.update(metric)
    return metrics


def compute_lcs_metrics(generated_outputs, target_outputs, threshold: float = 0.8):
    """
    LCS-based loose metric for triplet extraction.
    A predicted element is considered matched to target element if:
    - len(pred) < len(target): len(LCS(pred, target)) / len(target) > threshold
    - otherwise: target is a substring of pred
    This rule is applied to all 3 elements in each triplet.
    """

    def format_generated_output(preds, golds):
        task = {
            "ate": {"preds": [], "golds": []},
            "atsc": {"preds": [], "golds": []},
            "aspe": {"preds": [], "golds": []},
            "aooe": {"preds": [], "golds": []},
            "aope": {"preds": [], "golds": []},
            "aoste": {"preds": [], "golds": []}
        }
        for pred, gold in zip(preds, golds):
            if "[ate]" in gold:
                gold = gold.split("[ate]")[-1]
                pred = pred.split("[ate]")[-1]

                task["ate"]["golds"].append(gold)
                task["ate"]["preds"].append(pred)

            elif "[atsc]" in gold:
                gold = gold.split("[atsc]")[-1]
                pred = pred.split("[atsc]")[-1]

                task["atsc"]["golds"].append(gold)
                task["atsc"]["preds"].append(pred)

            elif "[aspe]" in gold:
                gold = gold.split("[aspe]")[-1]
                pred = pred.split("[aspe]")[-1]

                task["aspe"]["golds"].append(gold)
                task["aspe"]["preds"].append(pred)

            elif "[aooe]" in gold:
                gold = gold.split("[aooe]")[-1]
                pred = pred.split("[aooe]")[-1]

                task["aooe"]["golds"].append(gold)
                task["aooe"]["preds"].append(pred)

            elif "[aope]" in gold:
                gold = gold.split("[aope]")[-1]
                pred = pred.split("[aope]")[-1]

                task["aope"]["golds"].append(gold)
                task["aope"]["preds"].append(pred)

            elif "[aoste]" in gold:
                gold = gold.split("[aoste]")[-1]
                pred = pred.split("[aoste]")[-1]

                task["aoste"]["golds"].append(gold)
                task["aoste"]["preds"].append(pred)
        return task

    def normalize_text(value: str) -> str:
        return value.strip()

    def check_match_with_lcs(pred: str, target: str, lcs_threshold: float) -> bool:
        pred = normalize_text(pred)
        target = normalize_text(target)

        if pred == "" and target == "":
            return True
        if pred == "" or target == "":
            return False

        lcs_len = _longest_common_subsequence_length(pred, target)
        if len(pred) < len(target):
            return (lcs_len / len(target)) > lcs_threshold
        else:
            return (lcs_len / len(pred)) > lcs_threshold 

    def compute_metrics(preds, golds, prefix):
        total_pred = 0
        total_gt = 0
        tp = 0

        for gold, pred in zip(golds, preds):
            gold_list = [x.strip() for x in gold.split("##")]
            pred_list = [x.strip() for x in pred.split("##")]

            if len(gold_list) == 1 and gold_list[0] == "":
                gold_list = []
            if len(pred_list) == 1 and pred_list[0] == "":
                pred_list = []

            total_pred += len(pred_list)
            total_gt += len(gold_list)

            if prefix != "aoste":
                for gold_val in gold_list:
                    for pred_val in pred_list:
                        pred_val = pred_val.strip()
                        gold_val = gold_val.strip()
                        if pred_val in gold_val or gold_val in pred_val:
                            tp += 1
                            break
            else:
                for gold_val in gold_list:
                    gold_components = [x.strip() for x in gold_val.split("$")]
                    if len(gold_components) != 3:
                        continue

                    gold_aspect, gold_opinion, gold_sentiment = gold_components

                    for pred_val in pred_list:
                        pred_components = [x.strip() for x in pred_val.split("$")]
                        if len(pred_components) != 3:
                            continue

                        pred_aspect, pred_opinion, pred_sentiment = pred_components

                        if (
                            check_match_with_lcs(pred_aspect, gold_aspect, threshold)
                            and check_match_with_lcs(pred_opinion, gold_opinion, threshold)
                            and check_match_with_lcs(pred_sentiment, gold_sentiment, threshold)
                        ):
                            tp += 1
                            break

        p = tp / total_pred if total_pred > 0 else 0
        r = tp / total_gt if total_gt > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        return {
            f"{prefix}_precision": p,
            f"{prefix}_recall": r,
            f"{prefix}_f1": f1,
        }

    new_format = format_generated_output(generated_outputs, target_outputs)
    metrics = {}
    for task_name, value in new_format.items():
        metric = compute_metrics(**value, prefix=task_name)
        metrics.update(metric)
    return metrics

    