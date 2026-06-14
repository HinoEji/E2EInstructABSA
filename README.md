# Usage 
```{bash}
!python train.py --config config/config.json --set result_dir="./result" 

```

```{bash}

  python predict.py \
    --model-path result/best_models/eval_aoste_f1 \
    --dataset-path data/span/test.csv \
    --config config/config.json \
    --output-dir prediction_result
```

# Result folder structure
```
result
├── best_models/
├── predictions/
    ├── predict_<metric_name>.csv
    ├── metric_<metric_name>.json
    ├── ...

```