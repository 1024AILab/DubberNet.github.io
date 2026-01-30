import pandas as pd
import os
import shutil
from tqdm import tqdm

# =========================
# Path Configuration (PRID multi-shot)
# =========================
# PRID multi-shot directories for two cameras
cam1_path = r'xxx\multi_shot\cam_a'
cam2_path = r'xxx\multi_shot\cam_b'

# Common Voice TSV metadata and audio clips folder
tsv_path = r'xxx\train.tsv'
audio_clips_path = r'xxx\clips'

# Output directories where audio will be copied/renamed to match image names
cam1_audio_path = r'xxx\multi_shot\cam_a_audio'
cam2_audio_path = r'xxx\multi_shot\cam_b_audio'

# Create output audio folders if they do not exist
os.makedirs(cam1_audio_path, exist_ok=True)
os.makedirs(cam2_audio_path, exist_ok=True)

# =========================
# Load Common Voice metadata
# =========================
# Read TSV and get all client_id with their audio sample counts
audio_data = pd.read_csv(tsv_path, sep='\t')
audio_count = audio_data['client_id'].value_counts()

# Candidate speaker pool: all client_id (no threshold here)
selected_ids = audio_count.index.tolist()

# =========================
# Collect all person_ids from both cameras
# =========================
# person_ids: union of identities appearing in cam_a and cam_b
person_ids = set()
for cam_folder in [cam1_path, cam2_path]:
    person_ids.update([
        folder for folder in os.listdir(cam_folder)
        if os.path.isdir(os.path.join(cam_folder, folder))
    ])

# =========================
# Count images per person_id across both cameras
# =========================
# person_image_counts[person_id] = total number of png frames across cam_a + cam_b
person_image_counts = {}
for person_id in sorted(person_ids):
    image_count = 0
    for cam_folder in [cam1_path, cam2_path]:
        person_folder = os.path.join(cam_folder, person_id)
        if os.path.isdir(person_folder):
            images = [f for f in os.listdir(person_folder) if f.endswith('.png')]
            image_count += len(images)
    person_image_counts[person_id] = image_count

# =========================
# Assign one unique client_id (speaker) to each person_id
# =========================
# Goal: each person_id gets a unique client_id whose audio count is enough
# to cover required_audio_count = (#images // 6) + 1.
person_audio_map = {}   # person_id -> client_id
assigned_ids = set()    # already used client_id (enforce uniqueness across person_id)
available_audio = {}    # client_id -> list of remaining audio paths (to avoid immediate repeats)

for person_id in sorted(person_ids):
    # We allocate 1 audio per 6 images; +1 ensures enough when not divisible by 6
    required_audio_count = (person_image_counts[person_id] // 6) + 1

    assigned = False
    for selected_id in selected_ids:
        # Only consider speakers not yet assigned to any person_id
        if selected_id not in assigned_ids:
            client_audio_files = audio_data[audio_data['client_id'] == selected_id]['path'].values

            # Check if this speaker has enough audio clips for this person_id
            if len(client_audio_files) >= required_audio_count:
                person_audio_map[person_id] = selected_id
                assigned_ids.add(selected_id)
                available_audio[selected_id] = list(client_audio_files)  # initialize available pool
                assigned = True
                break

    # If we cannot find a qualified speaker for this person_id, stop early
    if not assigned:
        print(f"没有足够的client_id可分配给person_id {person_id}")
        break  # or continue to the next person_id, depending on your preference

# =========================
# Allocate/copy audio for cam_a and cam_b
# =========================
mapping_records = []  # rows for CSV traceability

for cam_folder, cam_audio_path in [(cam1_path, cam1_audio_path), (cam2_path, cam2_audio_path)]:
    for person_folder in tqdm(os.listdir(cam_folder), desc=f"处理 {cam_folder}"):
        person_path = os.path.join(cam_folder, person_folder)
        person_audio_folder = os.path.join(cam_audio_path, person_folder)

        # Only process identity folders
        if os.path.isdir(person_path):
            # Create mirrored output directory for this identity
            os.makedirs(person_audio_folder, exist_ok=True)

            # Sorted for deterministic allocation
            images = sorted([f for f in os.listdir(person_path) if f.endswith('.png')])
            person_id = person_folder

            # Ensure this person_id has been assigned a speaker
            if person_id in person_audio_map:
                selected_id = person_audio_map[person_id]
            else:
                print(f"person_id {person_id} 没有被分配client_id")
                continue  # skip this identity

            # If the available pool is empty, refill (this enables reuse after exhausting)
            if not available_audio[selected_id]:
                available_audio[selected_id] = list(
                    audio_data[audio_data['client_id'] == selected_id]['path'].values
                )

            # Allocate 1 audio clip per group of 6 images
            for i in range(0, len(images), 6):
                image_group = images[i:i + 6]

                if available_audio[selected_id]:
                    # Pick one unused audio file from the pool
                    audio_file = available_audio[selected_id].pop(0)
                    audio_file_path = os.path.join(audio_clips_path, audio_file)

                    # Copy audio file for each image and rename to <image_name>.mp3
                    for image in image_group:
                        image_name = os.path.splitext(image)[0]
                        new_audio_path = os.path.join(person_audio_folder, f"{image_name}.mp3")
                        shutil.copyfile(audio_file_path, new_audio_path)

                        # Record mapping for reproducibility/debugging
                        mapping_records.append([
                            image_name, person_id, selected_id,
                            f"{image_name}.mp3", audio_file
                        ])
                else:
                    # This should rarely happen due to our "required_audio_count" check
                    print(f"client_id {selected_id} 没有更多的音频文件可用")
                    break

# =========================
# Save mapping relationships to CSV
# =========================
mapping_df = pd.DataFrame(
    mapping_records,
    columns=['Image_Name', 'Person_ID', 'Client_ID', 'Audio_File_Name', 'Original_Audio_File']
)
mapping_df.to_csv('mapping_relationship.csv', index=False)

