from datasets import load_dataset, get_dataset_config_names
import pandas as pd, os

save_dir = os.path.expanduser("~/data/math")
os.makedirs(save_dir, exist_ok=True)

# 1️⃣  8 教科の config 一覧を取得
configs = get_dataset_config_names("EleutherAI/hendrycks_math")
# ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra',
#  'number_theory', 'prealgebra', 'precalculus', 'linear_algebra']

def load_concat(split):
    """全 config を読み込み、problem / solution だけ concat して返す DataFrame"""
    dfs = []
    for cfg in configs:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg, split=split)
        dfs.append(ds.to_pandas()[["problem", "solution"]])
    return pd.concat(dfs, ignore_index=True)

# 2️⃣  train / test を結合
train_df = load_concat("train")   # 7 500 行
test_df  = load_concat("test")    # 5 000 行

# 3️⃣  Parquet 保存
train_df.to_parquet(f"{save_dir}/train.parquet", index=False)
test_df.to_parquet(f"{save_dir}/test.parquet",  index=False)
print("✅ Saved:", save_dir, "-",
      train_df.shape, "train |", test_df.shape, "test")
