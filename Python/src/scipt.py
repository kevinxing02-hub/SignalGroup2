# sanity_check_loader.py
from load_preprocess import load_training_data
import pprint
import os

# --- 1. 指定任意一个 training 记录 ---
record_id = "R1"   # 你可以换成 R2、R3...
base = "S:/SignalGoupWork/training/"

edf = os.path.join(base, f"{record_id}.edf")
xml = os.path.join(base, f"{record_id}.xml")

print("=== Running Sanity Check on Loader ===")
print("EDF:", edf)
print("XML:", xml)

# --- 2. 调用你的 loader（带 canonical / target_fs 与否都行） ---
data, labels, channel_info = load_training_data(
    edf,
    xml,
    epoch_length=30,
    target_fs=None,        # 若你想测试 resample=125，可写 target_fs=125
    canonical_channel_lists=None
)

print("\n=== Channel Info ===")
pprint.pprint(channel_info)

# --- 3. 检查关键字段是否存在 ---
required_keys = [
    "eeg_fs", "eeg_samps_per_epoch",
    "eog_fs", "eog_samps_per_epoch",
    "emg_fs", "emg_samps_per_epoch"
]

print("\n=== Checking Channel Info Fields ===")
for key in required_keys:
    if key in channel_info:
        print(f"✓ {key} = {channel_info[key]}")
    else:
        print(f"✗ MISSING: {key}")

print("\n=== Shape Summary ===")
for k, v in data.items():
    print(f"{k.upper():5s}: {v.shape}")

print("\n=== Done. ===")
