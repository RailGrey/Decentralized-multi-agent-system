from datasets import load_dataset

ds_train = load_dataset(
    "deepmind/code_contests",
    cache_dir="dataset",
)