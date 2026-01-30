import pandas as pd
import os
import shutil
from tqdm import tqdm

# =========================
# Path Configuration (MARS)
# =========================
# MARS dataset directories
train_path = r'xxx\bbox_train'
test_path = r'xxx\Mars\bbox_test'

# Common Voice TSV + clips directory (can be multiple language packs or splits)
# Format: (tsv_path, clips_folder)
tsv_paths = [
    (r'xxx\en\train.tsv', r'xxx\clips'),
]

# Output directories for audio-augmented MARS
train_audio_path = r'xxx\Mars\bbox_train_audio'
test_audio_path = r'xxx\Mars\bbox_test_audio'

# Create output directories if they do not exist
os.makedirs(train_audio_path, exist_ok=True)
os.makedirs(test_audio_path, exist_ok=True)

# =========================
# Load Common Voice metadata and build full audio paths
# =========================
audio_data_list = []
for tsv_path, clips_path in tqdm(tsv_paths, desc="读取 csv 文件"):
    # Read TSV (Common Voice metadata)
    data = pd.read_csv(tsv_path, sep='\t', low_memory=False)

    # Store the clips base directory for each row
    data['audio_clips_path'] = clips_path

    # Build full file path: <clips_path>/<relative_audio_path_from_tsv>
    data['full_path'] = data.apply(
        lambda row: os.path.join(row['audio_clips_path'], row['path']),
        axis=1
    )

    audio_data_list.append(data)

# Merge all TSV sources into one dataframe
audio_data = pd.concat(audio_data_list, ignore_index=True)

# =========================
# Build mapping: client_id -> list of audio file full paths
# =========================
client_audio_map = {}
for client_id, group in tqdm(audio_data.groupby('client_id'), desc="构建 client_id 到音频文件列表的映射"):
    client_audio_map[client_id] = group['full_path'].tolist()

# Sort speakers by number of available audio clips (descending)
# This makes it easier to satisfy larger identities first.
sorted_client_ids = sorted(
    client_audio_map.keys(),
    key=lambda x: len(client_audio_map[x]),
    reverse=True
)

# =========================
# Utility: collect all images under a person_id folder
# =========================
def get_images_in_person_folder(person_folder_path):
    """
    Collect all image file paths under the given person folder (recursive).
    Returns:
      images: list of tuples (absolute_image_path, relative_path_to_person_folder)
    """
    images = []
    for root, dirs, files in os.walk(person_folder_path):
        for f in files:
            if f.endswith('.png') or f.endswith('.jpg'):
                image_path = os.path.join(root, f)

                # Store the relative path with respect to person_folder_path
                relative_path = os.path.relpath(image_path, person_folder_path)
                images.append((image_path, relative_path))
    return images

# =========================
# Collect person_id list for train/test
# =========================
train_person_ids = [
    folder for folder in os.listdir(train_path)
    if os.path.isdir(os.path.join(train_path, folder))
]
test_person_ids = [
    folder for folder in os.listdir(test_path)
    if os.path.isdir(os.path.join(test_path, folder))
]

# =========================
# Count images per person_id (train)
# =========================
train_person_image_counts = {}
for person_id in tqdm(sorted(train_person_ids), desc="统计 train 中每个 person_id 的图片数量"):
    person_folder = os.path.join(train_path, person_id)
    images = get_images_in_person_folder(person_folder)
    train_person_image_counts[person_id] = len(images)

# =========================
# Count images per person_id (test)
# =========================
test_person_image_counts = {}
for person_id in tqdm(sorted(test_person_ids), desc="统计 test 中每个 person_id 的图片数量"):
    person_folder = os.path.join(test_path, person_id)
    images = get_images_in_person_folder(person_folder)
    test_person_image_counts[person_id] = len(images)

# =========================
# Assign unique client_id to each person_id (train + test)
# =========================
# person_audio_map: person_id -> client_id
# assigned_ids: to ensure one speaker is used by at most one identity
# available_audio: client_id -> remaining audio pool (copied list)
person_audio_map = {}
assigned_ids = set()
available_audio = {}

# ---------
# Assign for train identities
# ---------
for person_id in tqdm(sorted(train_person_ids), desc="为 train 中的 person_id 分配 client_id"):
    # Allocate 1 audio clip per 6 images; +1 ensures enough if not divisible by 6
    required_audio_count = (train_person_image_counts[person_id] // 6) + 1

    assigned = False
    for selected_id in sorted_client_ids:
        # Pick an unassigned speaker with enough audio clips
        if selected_id not in assigned_ids and len(client_audio_map[selected_id]) >= required_audio_count:
            person_audio_map[person_id] = selected_id
            assigned_ids.add(selected_id)

            # Copy audio list for consumption (pop) later
            available_audio[selected_id] = client_audio_map[selected_id][:]
            assigned = True
            break

    if not assigned:
        print(f"没有足够的client_id可分配给 person_id {person_id}")
        continue  # skip this identity

# ---------
# Assign for test identities
# ---------
for person_id in tqdm(sorted(test_person_ids), desc="为 test 中的 person_id 分配 client_id"):
    required_audio_count = (test_person_image_counts[person_id] // 6) + 1

    assigned = False
    for selected_id in sorted_client_ids:
        if selected_id not in assigned_ids and len(client_audio_map[selected_id]) >= required_audio_count:
            person_audio_map[person_id] = selected_id
            assigned_ids.add(selected_id)
            available_audio[selected_id] = client_audio_map[selected_id][:]
            assigned = True
            break

    if not assigned:
        print(f"没有足够的client_id可分配给 person_id {person_id}")
        continue  # skip this identity

# =========================
# Allocate/copy audio into train/test audio folders
# =========================
mapping_records = []  # rows for CSV traceability

# ---------
# Process train split
# ---------
for person_id in tqdm(train_person_ids, desc="处理 train"):
    person_path = os.path.join(train_path, person_id)

    # Collect all images (abs path + relative path)
    images = get_images_in_person_folder(person_path)
    images.sort(key=lambda x: x[1])  # sort by relative path for deterministic ordering

    # Ensure this identity has an assigned speaker
    if person_id in person_audio_map:
        selected_id = person_audio_map[person_id]
    else:
        print(f"person_id {person_id} 没有被分配 client_id")
        continue

    # If audio pool is empty, refill from the original list (allows reuse after exhaustion)
    if not available_audio[selected_id]:
        available_audio[selected_id] = client_audio_map[selected_id][:]

    # Allocate 1 audio clip per 6 images; copy same audio to all images in the group
    for i in range(0, len(images), 6):
        image_group = images[i:i + 6]

        if available_audio[selected_id]:
            audio_file_path = available_audio[selected_id].pop(0)

            for image_path, relative_path in image_group:
                image_name = os.path.splitext(os.path.basename(image_path))[0]

                # Preserve MARS sub-folder structure by mirroring relative image path directories
                audio_output_path = os.path.join(train_audio_path, person_id, os.path.dirname(relative_path))
                os.makedirs(audio_output_path, exist_ok=True)

                new_audio_path = os.path.join(audio_output_path, f"{image_name}.mp3")
                shutil.copyfile(audio_file_path, new_audio_path)

                # Record mapping information for reproducibility
                mapping_records.append([
                    relative_path, person_id, selected_id,
                    new_audio_path, audio_file_path, 'train'
                ])
        else:
            print(f"client_id {selected_id} 没有更多的音频文件可用")
            break

# ---------
# Process test split
# ---------
for person_id in tqdm(test_person_ids, desc="处理 test"):
    person_path = os.path.join(test_path, person_id)

    images = get_images_in_person_folder(person_path)
    images.sort(key=lambda x: x[1])

    if person_id in person_audio_map:
        selected_id = person_audio_map[person_id]
    else:
        print(f"person_id {person_id} 没有被分配 client_id")
        continue

    if not available_audio[selected_id]:
        available_audio[selected_id] = client_audio_map[selected_id][:]

    for i in range(0, len(images), 6):
        image_group = images[i:i + 6]

        if available_audio[selected_id]:
            audio_file_path = available_audio[selected_id].pop(0)

            for image_path, relative_path in image_group:
                image_name = os.path.splitext(os.path.basename(image_path))[0]

                audio_output_path = os.path.join(test_audio_path, person_id, os.path.dirname(relative_path))
                os.makedirs(audio_output_path, exist_ok=True)

                new_audio_path = os.path.join(audio_output_path, f"{image_name}.mp3")
                shutil.copyfile(audio_file_path, new_audio_path)

                mapping_records.append([
                    relative_path, person_id, selected_id,
                    new_audio_path, audio_file_path, 'test'
                ])
        else:
            print(f"client_id {selected_id} 没有更多的音频文件可用")
            break

