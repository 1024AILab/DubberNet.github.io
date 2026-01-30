import pandas as pd
import os
import shutil
from tqdm import tqdm

# =========================
# Path Configuration
# =========================
# Root directories for iLIDS-VID image sequences (two cameras)
cam1_path = r'xxx\sequences\cam1'
cam2_path = r'xxx\sequences\cam2'

# Common Voice TSV (metadata) and clips directory (audio files)
tsv_path = r'xxx\train.tsv'
audio_clips_path = r'xxx\clips'

# Output directories: audio files copied/assigned to match iLIDS-VID frame names
cam1_audio_path = r'xxx\sequences\cam1_audio'
cam2_audio_path = r'xxx\sequences\cam2_audio'

# Create output audio folders if they do not exist
os.makedirs(cam1_audio_path, exist_ok=True)
os.makedirs(cam2_audio_path, exist_ok=True)

# =========================
# Load Common Voice metadata and select qualified speakers
# =========================
# Read TSV and select client_id with more than 52 audio samples
audio_data = pd.read_csv(tsv_path, sep='\t')
audio_count = audio_data['client_id'].value_counts()
selected_ids = audio_count[audio_count > 52].index.tolist()

# person_audio_map: maps each person_id (identity) -> a unique client_id (speaker)
person_audio_map = {}

# assigned_ids: tracks client_id already assigned to some person_id (enforce uniqueness)
assigned_ids = set()

# mapping_records: rows to export into CSV for traceability
# Each row: [Image_Name, Client_ID, Audio_File_Name, Original_Audio_File]
mapping_records = []

# available_audio: per client_id, track remaining unused audio paths
# This prevents reusing the same audio file until the list is exhausted.
available_audio = {}

# =========================
# Iterate both cameras and allocate audio
# =========================
# For each camera folder, we create an audio folder mirroring the person_id subfolders.
for cam_folder, cam_audio_path in [(cam1_path, cam1_audio_path), (cam2_path, cam2_audio_path)]:
    for person_folder in tqdm(os.listdir(cam_folder), desc="Processing"):
        person_path = os.path.join(cam_folder, person_folder)
        person_audio_folder = os.path.join(cam_audio_path, person_folder)

        # Only process directories (each directory is one identity/person_id)
        if os.path.isdir(person_path):
            # Create a matching output directory for this person_id
            os.makedirs(person_audio_folder, exist_ok=True)

            # Collect all PNG frames in this identity folder (sorted for deterministic mapping)
            images = sorted([f for f in os.listdir(person_path) if f.endswith('.png')])

            # In iLIDS-VID, folder name typically corresponds to person_id
            person_id = person_folder

            # -------------------------
            # Step A: Assign a unique client_id to this person_id (only once globally)
            # -------------------------
            # If this is the first time we see this person_id, assign a new (unused) client_id.
            # Note: although the comment says "in cam1", the code is actually global across both cams.
            if person_id not in person_audio_map and selected_ids:
                while selected_ids:
                    # Pop one qualified client_id from the list
                    selected_id = selected_ids.pop(0)

                    # Ensure each client_id is assigned to at most one person_id
                    if selected_id not in assigned_ids:
                        # List all audio file paths belonging to this client_id
                        audio_paths = list(audio_data[audio_data['client_id'] == selected_id]['path'].values)

                        # Initialize available audio pool for this speaker
                        available_audio[selected_id] = audio_paths.copy()

                        # Bind this person_id to this client_id
                        person_audio_map[person_id] = selected_id
                        assigned_ids.add(selected_id)
                        break  # assignment successful

            # -------------------------
            # Step B: For this person_id (both cam1 and cam2), reuse the same client_id
            # -------------------------
            if person_id in person_audio_map:
                selected_id = person_audio_map[person_id]

                # If we have consumed all audio files for this client_id, refill the list (allow reuse)
                if not available_audio[selected_id]:
                    available_audio[selected_id] = list(
                        audio_data[audio_data['client_id'] == selected_id]['path'].values
                    )

                # -------------------------
                # Step C: Allocate one audio file per group of 6 frames
                # -------------------------
                # iLIDS-VID sequences often have multiple frames; here we assign 1 audio file
                # to every consecutive block of 6 frames, and copy it for each frame in that block.
                for i in range(0, len(images), 6):
                    image_group = images[i:i + 6]

                    if available_audio[selected_id]:
                        # Take one unused audio file from this speaker
                        audio_file = available_audio[selected_id].pop(0)

                        # Full path to the audio clip in Common Voice clips directory
                        audio_file_path = os.path.join(audio_clips_path, audio_file)

                        # Copy and rename audio file to match each frame name: <frame>.mp3
                        for image in image_group:
                            image_name = os.path.splitext(image)[0]  # frame name without extension
                            new_audio_path = os.path.join(person_audio_folder, f"{image_name}.mp3")
                            shutil.copyfile(audio_file_path, new_audio_path)

                            # Record mapping for debugging and reproducibility
                            mapping_records.append([image_name, selected_id, f"{image_name}.mp3", audio_file])

# =========================
# Save mapping file
# =========================
mapping_df = pd.DataFrame(
    mapping_records,
    columns=['Image_Name', 'Client_ID', 'Audio_File_Name', 'Original_Audio_File']
)
mapping_df.to_csv('mapping_relationship.csv', index=False)
print("Mapping has been saved to mapping_relationship.csv")

