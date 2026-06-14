import pandas as pd
from pathlib import Path

# Cấu hình
SOURCE_DIR = Path("data/v3nobrand")
TARGET_DIR = Path("data/v3nobrand_small")
SAMPLE_SIZE_TRAIN = 150  # Số samples từ train.csv
SAMPLE_SIZE_DEV = 50     # Số samples từ dev.csv
SAMPLE_SIZE_TEST = 50    # Số samples từ test.csv

# Tạo thư mục target
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# Xử lý train.csv
print("Processing train.csv...")
train_df = pd.read_csv(SOURCE_DIR / "train.csv")
train_small = train_df.sample(n=min(SAMPLE_SIZE_TRAIN, len(train_df)), random_state=42)
train_small.to_csv(TARGET_DIR / "train.csv", index=False)
print(f"✓ Saved {len(train_small)} samples to train.csv")

# Xử lý dev.csv
print("Processing dev.csv...")
dev_df = pd.read_csv(SOURCE_DIR / "dev.csv")
dev_small = dev_df.sample(n=min(SAMPLE_SIZE_DEV, len(dev_df)), random_state=42)
dev_small.to_csv(TARGET_DIR / "dev.csv", index=False)
print(f"✓ Saved {len(dev_small)} samples to dev.csv")

# Xử lý test.csv
print("Processing test.csv...")
test_df = pd.read_csv(SOURCE_DIR / "test.csv")
test_small = test_df.sample(n=min(SAMPLE_SIZE_TEST, len(test_df)), random_state=42)
test_small.to_csv(TARGET_DIR / "test.csv", index=False)
print(f"✓ Saved {len(test_small)} samples to test.csv")

# Xử lý aos_nobrand.csv nếu có
if (SOURCE_DIR / "aos_nobrand.csv").exists():
    print("Processing aos_nobrand.csv...")
    aos_df = pd.read_csv(SOURCE_DIR / "aos_nobrand.csv")
    aos_small = aos_df.sample(n=min(100, len(aos_df)), random_state=42)
    aos_small.to_csv(TARGET_DIR / "aos_nobrand.csv", index=False)
    print(f"✓ Saved {len(aos_small)} samples to aos_nobrand.csv")

# Sao chép example folder
import shutil
example_source = SOURCE_DIR / "example"
example_target = TARGET_DIR / "example"
if example_source.exists():
    if example_target.exists():
        shutil.rmtree(example_target)
    shutil.copytree(example_source, example_target)
    print("✓ Copied example folder")

print("\n✅ Hoàn tất! Dataset nhỏ được tạo tại:", TARGET_DIR)
print("   - train.csv: {} samples".format(len(train_small)))
print("   - dev.csv: {} samples".format(len(dev_small)))
print("   - test.csv: {} samples".format(len(test_small)))
