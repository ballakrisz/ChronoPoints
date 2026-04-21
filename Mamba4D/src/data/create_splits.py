import numpy as np
import os
import re
import json
from collections import defaultdict
import random
from collections import defaultdict

OUTPUT_DIR = "/home/appuser/drone_tracking/tmux/time_series_classification/data/dataset/train_test_split/"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'train_test_split_v1/')
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure the folder exists

SINGLE_UAV_DATASET_PATH = os.path.join(os.path.dirname(__file__), '02958343/single_return/')
DOUBLE_UAV_DATASET_PATH = os.path.join(os.path.dirname(__file__), '02958343/double_return/')
BIRD_DATASET_PATH = os.path.join(os.path.dirname(__file__), '01503061/')

# TODO: MIN 10 valid frames are needed, then keep them tight (eg.: no 5 pad frames consecutively)

def sort_key(path):
    match = re.search(r'obj(\d+)_mov(\d+)', path)
    if match:
        obj_num = int(match.group(1))
        mov_num = int(match.group(2))
        return (obj_num, mov_num)
    return (float('inf'), float('inf'))  # fallback for unmatched paths

def extract_timestamp(file_name):
    match = re.search(r'frame_(\d+)\.npz$', file_name)
    return int(match.group(1)) if match else 0

def load_and_pad_sequence(movement_path, frame_interval=100_000_000, pad_token="<PAD/>", min_frames=2, uav=False):

    if uav:
        files = [
            '/'.join(os.path.join(movement_path, f).split('/')[-4:])
            for f in os.listdir(movement_path)
            if f.endswith('.npz')
        ]
    else:
        files = [
            '/'.join(os.path.join(movement_path, f).split('/')[-3:])
            for f in os.listdir(movement_path)
            if f.endswith('.npz')
        ]

    # skip sequences with less than [min_frames] valid frames
    if len(files) < min_frames:
        return None

    # Sort files by timestamp
    files.sort(key=lambda f: extract_timestamp(os.path.basename(f)))

    padded_sequence = []
    prev_ts = None

    for f in files:
        ts = extract_timestamp(os.path.basename(f))
        if prev_ts is not None:
            gap = ts - prev_ts
            rounded_gap = int((gap + frame_interval / 2) // frame_interval) * frame_interval
            num_missing = (rounded_gap // frame_interval) - 1
            padded_sequence.extend([pad_token] * num_missing)
        padded_sequence.append(f)
        prev_ts = ts

    return padded_sequence

def sliding_windows_from_padded_sequence(padded_seq, type_counter, max_len=5, min_frames=2 , pad_token="<PAD/>"):
    all_windows = []
    n = len(padded_seq)
    
    # get the label for the sequence
    for seq in padded_seq:
        if seq != pad_token:
            data = np.load(os.path.join(os.path.dirname(__file__), seq))
            label = data['type'].item()
            break

    for window_size in range(min_frames, max_len + 1):
        for start_idx in range(n - window_size + 1):
            window = padded_seq[start_idx:start_idx + window_size]

            # Skip windows starting with pad: to make it realistic, we only run the model when a new samples arrives
            # Skip windows ending with pad: to avoid duplicates; If the samples are: f x f x x 
            # 1. pass: [f] --> f x x x x
            # 2. pass: [f, x] --> f x x x
            # They produce the same output, which is redundant
            if window[0] == pad_token or window[-1] == pad_token:
                continue

            # Pad window up to max_len
            if len(window) < max_len:
                window = list(window) + [pad_token] * (max_len - len(window))
            
            # Count valid (non-pad) frames
            valid_indices = [i for i, x in enumerate(window) if x != pad_token]
            num_valid = len(valid_indices)

            # only keep windows with at least [min_frames-1] valid frames -> we can afford to have 1 pad token in between
            if num_valid < min_frames - 1:
                continue 

            # make sure the 2 pad tokens are not next to each other
            if num_valid == min_frames - 2:
                skip = False
                # Check distance between the two valid indices
                for i in range(1, len(valid_indices)):
                    if valid_indices[i] - valid_indices[i-1] > 2:
                        skip = True
                        break
                if skip:
                    continue

            all_windows.append(window)
            type_counter[label] += 1
    # Double check to make sure there are no duplicate (same) windows
    seen = set()
    duplicates = 0
    for w in all_windows:
        tup = tuple(w)  # lists can’t go in a set
        if tup in seen:
            duplicates += 1
        else:
            seen.add(tup)
    if duplicates > 0:
        print(f"Duplicate windows found: {duplicates}")
    
    return all_windows, type_counter

def shuffle_and_split(data, train_ratio=0.7, val_ratio=0.15, seed=42):
    random.seed(seed)
    random.shuffle(data)
    
    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]

    return train, val, test

def save_splits(prefix, dataset):
    train, val, test = shuffle_and_split(dataset)

    dst_dir = os.path.join(OUTPUT_DIR, prefix)
    os.makedirs(dst_dir, exist_ok=True)  # Ensure the folder exists

    with open(os.path.join(dst_dir, f"shuffled_train_file_list.json"), "w") as f:
        json.dump(train, f, indent=2)

    with open(os.path.join(dst_dir, f"shuffled_val_file_list.json"), "w") as f:
        json.dump(val, f, indent=2)

    with open(os.path.join(dst_dir, f"shuffled_test_file_list.json"), "w") as f:
        json.dump(test, f, indent=2)


single_birds = [os.path.join(BIRD_DATASET_PATH, d) for d in os.listdir(BIRD_DATASET_PATH) if os.path.isdir(os.path.join(BIRD_DATASET_PATH, d))]
single_uavs = [os.path.join(SINGLE_UAV_DATASET_PATH, d) for d in os.listdir(SINGLE_UAV_DATASET_PATH) if os.path.isdir(os.path.join(SINGLE_UAV_DATASET_PATH, d))]
double_uavs = [os.path.join(DOUBLE_UAV_DATASET_PATH, d) for d in os.listdir(DOUBLE_UAV_DATASET_PATH) if os.path.isdir(os.path.join(DOUBLE_UAV_DATASET_PATH, d))]

single_birds.sort(key=sort_key)
single_uavs.sort(key=sort_key)
double_uavs.sort(key=sort_key)

# legalabbb 2 frame, max 1 db lyuk lehet kozottuk, legyen parametrikus, de a legalabb frame is legyen parametrikus
# make_shifts param -> ha be van allitva, akkor 2 frame eseten is kitoljuk azt: f f x x x -> x f f x x  -> x x f f x
# make them the same, so we only have fixed lengths of frames
min_frames = 10
max_frames = 10
shift_empty = True

# collect number of samples per type
type_counter = defaultdict(int)

all_single_birds = []
for bird in single_birds:
    filled = load_and_pad_sequence(bird, min_frames=min_frames, uav=False)
    if filled is None:
        continue
    padded, type_counter = sliding_windows_from_padded_sequence(filled, type_counter, max_len=max_frames, min_frames=min_frames, pad_token="<PAD/>")
    all_single_birds.extend(padded)

all_single_uavs = []
for uav in single_uavs:
    filled = load_and_pad_sequence(uav, min_frames=min_frames, uav=True)
    if filled is None:
        continue
    padded, type_counter = sliding_windows_from_padded_sequence(filled, type_counter, max_len=max_frames, min_frames=min_frames, pad_token="<PAD/>")
    all_single_uavs.extend(padded)

# all_double_uavs = []
# for uav in double_uavs:
#     filled = load_and_pad_sequence(uav, min_frames=min_frames, uav=True)
#     if filled is None:
#         continue
#     padded = sliding_windows_from_padded_sequence(filled, max_len=max_frames, min_frames=min_frames, pad_token="<PAD/>")
#     all_double_uavs.extend(padded)

# with open(os.path.join(OUTPUT_DIR, "single_return_birds.json"), "w") as f:
#     json.dump(all_single_birds, f, indent=2)

# with open(os.path.join(OUTPUT_DIR, "single_return_uavs.json"), "w") as f:
#     json.dump(all_single_uavs, f, indent=2)

# with open(os.path.join(OUTPUT_DIR, "double_return_uavs.json"), "w") as f:
#     json.dump(all_double_uavs, f, indent=2)


single_return_data = all_single_birds + all_single_uavs
# mixed_return_data = single_return_data + all_double_uavs

print(f"Single return samples: {len(single_return_data)}")
print("Sample counts per type:", dict(type_counter))

save_splits("single_return", single_return_data)
# save_splits("mixed_return", mixed_return_data)


