# Usage 
```{bash}
!python train.py --config config/config.json --set result_dir="./result" 

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