"""The dataloader for Aya's custom rPPG dataset.

=== Key Modifications (v3) ===
1. SUBJECT-LEVEL SPLIT: Uses explicit subject ID lists instead of BEGIN/END ratio.
   Controlled via YAML: TRAIN_SUBJECTS, TEST_SUBJECTS
2. CONDITION FILTERING: Filters sessions by distance, motion, lux via config lists.
   Controlled in this file: TRAIN_CONDITIONS / TEST_CONDITIONS dicts.
3. MOTION-LEVEL METADATA: Each preprocessed chunk stores its motion_level
   (e.g., "Stationary", "Talking") in a side-car .npy file for granular evaluation.
4. MOTION CSV INTEGRATION: Loads Patrick's motion CSVs from raw session folders
   and saves 8-channel motion feature npy files alongside input/label npy files.
   Channels: [head_tx_vel, head_ty_vel, lips_vel, yaw, pitch, roll, lips_accel, eor_l]
5. ALIGNMENT FIX: Uses nearest-neighbor GT alignment instead of direct searchsorted.
6. TRIM FIX: Proportional trim (2.5% each end) instead of hardcoded 100 samples.

Dataset Structure:
    DataRoot/
    |-- 0/                              <-- Subject folder (0~10)
    |   |-- vid_1m_30lux_Stationary_45deg_useiPhone/
    |   |   |-- 1709812345000.png       <-- Frame (filename = timestamp ms)
    |   |   |-- vid_1m_30lux_Stationary_45deg_useiPhone_pose.csv
    |   |   |-- vid_1m_30lux_Stationary_45deg_useiPhone_lips.csv
    |   |   |-- vid_1m_30lux_Stationary_45deg_useiPhone_left_eye.csv
    |   |   |-- vid_1m_30lux_Stationary_45deg_useiPhone_right_eye.csv
    |   |   |-- vid_1m_30lux_Stationary_45deg_useiPhone_left_cheek.csv
    |   |   |-- vid_1m_30lux_Stationary_45deg_useiPhone_right_cheek.csv
    |   |-- gt_1m_30lux_Stationary_45deg_useiPhone/
    |   |   |-- pulse_data.csv          <-- Columns: PC_Timestamp_ms, Signal_Value
    |-- 1/
    |-- ...
    |-- 10/
"""
import glob
import json
import os

import cv2
import numpy as np
import pandas as pd
from dataset.data_loader.BaseLoader import BaseLoader

# ============================================================================
#  CONFIGURATION: Edit these to control which sessions are loaded
# ============================================================================

# --- Subject Split ---
TRAIN_SUBJECTS = [1, 2, 3, 6, 8, 9, 10]
TEST_SUBJECTS  = [4, 5, 7]

# --- Train Conditions ---
TRAIN_CONDITIONS = {
    "distances": [1, 2, 3],
    "motions":   ['Nodding'],
    "lux":       [70],
}

# --- Test Conditions ---
TEST_CONDITIONS = {
    "distances": [1, 2, 3],
    "motions":   ['Nodding'],
    "lux":       [70],
}

# --- Motion feature channel order (must match DeepPhysMultiTask_v1 expectation) ---
# [head_tx_vel, head_ty_vel, lips_vel, yaw, pitch, roll, lips_accel, eor_l]
MOTION_N_CHANNELS = 8


def _parse_folder_name(folder_name):
    """Parse a vid/gt folder name into its condition components.

    Example: 'vid_1m_30lux_Stationary_45deg_useiPhone'
    Returns: {'distance': 1, 'lux': 30, 'motion': 'Stationary', 'angle': '45deg', 'device': 'useiPhone'}
    """
    if folder_name.startswith("vid_"):
        core = folder_name[4:]
    elif folder_name.startswith("gt_"):
        core = folder_name[3:]
    else:
        core = folder_name

    parts = core.split("_")
    result = {}
    try:
        result["distance"] = int(parts[0].replace("m", ""))
        result["lux"] = int(parts[1].replace("lux", ""))
        result["motion"] = parts[2]
        result["angle"] = parts[3] if len(parts) > 3 else ""
        result["device"] = parts[4] if len(parts) > 4 else ""
    except (IndexError, ValueError):
        result["distance"] = -1
        result["lux"] = -1
        result["motion"] = "Unknown"
        result["angle"] = ""
        result["device"] = ""
    return result


def _matches_conditions(parsed, conditions):
    """Check if a parsed folder matches the given condition filters."""
    if conditions.get("distances"):
        if parsed["distance"] not in conditions["distances"]:
            return False
    if conditions.get("motions"):
        if parsed["motion"] not in conditions["motions"]:
            return False
    if conditions.get("lux"):
        if parsed["lux"] not in conditions["lux"]:
            return False
    return True


class AyaLoader(BaseLoader):
    """The data loader for Aya's custom rPPG dataset (v3 with motion CSV integration)."""

    def __init__(self, name, data_path, config_data, device=None):
        self.split_name = name
        self.chunk_motion_map = {}
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Scan dataset root and return all valid vid/gt pairs with metadata."""
        if self.split_name in ("train",):
            allowed_subjects = TRAIN_SUBJECTS
            conditions = TRAIN_CONDITIONS
        else:
            allowed_subjects = TEST_SUBJECTS
            conditions = TEST_CONDITIONS

        data_dirs = []
        subject_folders = sorted(
            [d for d in os.listdir(data_path)
             if os.path.isdir(os.path.join(data_path, d)) and d.isdigit()],
            key=lambda x: int(x)
        )
        if not subject_folders:
            raise ValueError(self.dataset_name + " data paths empty!")

        trial_counter = 0
        for subj_folder in subject_folders:
            subject = int(subj_folder)
            if subject not in allowed_subjects:
                continue

            subj_path = os.path.join(data_path, subj_folder)
            vid_folders = sorted(glob.glob(os.path.join(subj_path, "vid_*")))

            for vid_folder in vid_folders:
                vid_name = os.path.basename(vid_folder)
                parsed = _parse_folder_name(vid_name)

                if not _matches_conditions(parsed, conditions):
                    continue

                gt_name = "gt_" + vid_name[4:]
                gt_folder = os.path.join(subj_path, gt_name)

                if os.path.isdir(gt_folder):
                    data_dirs.append({
                        "index": trial_counter,
                        "path": vid_folder,
                        "subject": subject,
                        "gt_path": gt_folder,
                        "motion_level": parsed["motion"],
                        "distance": parsed["distance"],
                        "lux": parsed["lux"],
                    })
                    trial_counter += 1

        if not data_dirs:
            raise ValueError(
                f"{self.dataset_name} [{self.split_name}] no valid vid/gt pairs found! "
                f"Subjects={allowed_subjects}, Conditions={conditions}"
            )

        print(f"\n[AyaLoader] Split='{self.split_name}', "
              f"Subjects={allowed_subjects}, "
              f"Sessions found={len(data_dirs)}")
        for d in data_dirs:
            print(f"  idx={d['index']} subj={d['subject']} "
                  f"motion={d['motion_level']} dist={d['distance']}m "
                  f"lux={d['lux']} -> {os.path.basename(d['path'])}")
        print()

        return data_dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Override: subject filtering already done in get_raw_data."""
        return data_dirs

    def multi_process_manager(self, data_dirs, config_preprocess, multi_process_quota=2):
        """Override: limit parallel processes."""
        return super().multi_process_manager(data_dirs, config_preprocess, multi_process_quota=2)

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Preprocess one session: read frames, align GT, crop face, chunk, save."""
        saved_filename = data_dirs[i]['index']
        vid_path = data_dirs[i]['path']
        gt_path = data_dirs[i]['gt_path']
        motion_level = data_dirs[i]['motion_level']
        print(f"[{i}] START", flush=True)
        print(f"\n[{i + 1}/{len(data_dirs)}] Processing: {os.path.basename(vid_path)} "
              f"(motion={motion_level})")

        # ---- Step 1: Read & trim ground truth ----
        gt_csv = os.path.join(gt_path, "pulse_data.csv")
        if not os.path.exists(gt_csv):
            print(f"[WARN] GT not found, skipping: {gt_csv}")
            file_list_dict[i] = []
            return
        print(f"[{i}] GT found", flush=True)

        gt_df = pd.read_csv(gt_csv)
        print(f"[{i}] GT loaded, rows={len(gt_df)}", flush=True)
        gt_df = gt_df[["PC_Timestamp_ms", "Signal_Value"]].dropna()

        # Proportional trim: 2.5% each end, minimum 10 samples (Fangshi fix)
        trim = max(10, int(len(gt_df) * 0.025))
        gt_df = gt_df.iloc[trim:-trim].reset_index(drop=True)
        print(f"[{i}] GT trimmed (trim={trim}), rows={len(gt_df)}", flush=True)

        if len(gt_df) < 30:
            print(f"[WARN] GT too short after trim, skipping: {gt_csv}")
            file_list_dict[i] = []
            return

        gt_timestamps = gt_df["PC_Timestamp_ms"].values.astype(np.float64)
        gt_signal = gt_df["Signal_Value"].values.astype(np.float64)

        # ---- Step 2: Get filenames & timestamps ----
        all_png = sorted(glob.glob(vid_path + os.sep + '*.png'))
        frame_ts = np.array([
            float(os.path.splitext(os.path.basename(p))[0]) for p in all_png
        ])

        # ---- Step 3: Filter to GT time range ----
        gt_start, gt_end = gt_timestamps[0], gt_timestamps[-1]
        mask = (frame_ts >= gt_start) & (frame_ts <= gt_end)
        selected_png = [p for p, m in zip(all_png, mask) if m]
        frame_ts = frame_ts[mask]

        if len(selected_png) < 30:
            print(f"[WARN] Too few frames after alignment, skipping: {vid_path}")
            file_list_dict[i] = []
            return

        # ---- Step 4: Align GT signal to frame timestamps (nearest-neighbor fix) ----
        indices = np.searchsorted(gt_timestamps, frame_ts, side='left')
        indices = np.clip(indices, 0, len(gt_timestamps) - 1)
        indices_left = np.clip(indices - 1, 0, len(gt_timestamps) - 1)
        left_dist = np.abs(gt_timestamps[indices_left] - frame_ts)
        right_dist = np.abs(gt_timestamps[indices] - frame_ts)
        indices = np.where(left_dist < right_dist, indices_left, indices)
        bvps = gt_signal[indices]

        print(f"[{i}] Reading frames...", flush=True)

        # ---- Step 5: Read frames ----
        frames = self.read_video_list(selected_png, resize_w=640, resize_h=360)
        print(f"[{i}] Frames read: {len(frames)}", flush=True)

        # ---- Step 6: Preprocess (face crop, normalize, chunk) & Save ----
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        print(f"[{i}] Preprocessing done, clips={frames_clips.shape[0]}", flush=True)
        input_name_list, label_name_list = self.save_multi_process(
            frames_clips, bvps_clips, saved_filename)

        # ---- Step 7: Load and save motion feature clips ----
        print(f"[{i}] Loading motion CSVs...", flush=True)
        motion_clips = self._load_motion_clips(
            vid_folder=vid_path,
            total_frames=len(selected_png),
            num_clips=frames_clips.shape[0],
            chunk_length=config_preprocess.CHUNK_LENGTH,
        )
        if motion_clips is not None:
            self._save_motion_clips(motion_clips, saved_filename)
            print(f"[{i}] Motion clips saved: {motion_clips.shape[0]}", flush=True)
        else:
            print(f"[{i}] Motion CSVs not found — motion features will be zeros at runtime.",
                  flush=True)

        # ---- Step 8: Save motion_level metadata ----
        print(f"[{i}] Saving metadata...", flush=True)
        motion_map = {}
        for inp_path in input_name_list:
            chunk_basename = os.path.basename(inp_path)
            motion_map[chunk_basename] = motion_level

        meta_path = os.path.join(
            self.cached_path,
            f"{saved_filename}_motion_meta.json"
        )
        with open(meta_path, 'w') as f:
            json.dump(motion_map, f)

        file_list_dict[i] = input_name_list

    # =========================================================================
    # MOTION CSV INTEGRATION
    # =========================================================================

    def _load_motion_clips(self, vid_folder, total_frames, num_clips, chunk_length):
        """
        Load 8-channel motion features from Patrick's CSVs and slice into clips.

        Uses pose CSV (yaw/pitch/roll), lips CSV (vel/accel magnitudes),
        and left/right eye CSVs (head translation proxy + eye openness proxy).

        Channels:
            0: head_tx_vel  — gradient of left-eye centroid x (head translation proxy)
            1: head_ty_vel  — gradient of left-eye centroid y
            2: lips_vel     — L2 magnitude of lips velocity xyz
            3: yaw          — head yaw from pose CSV
            4: pitch        — head pitch from pose CSV
            5: roll         — head roll from pose CSV
            6: lips_accel   — L2 magnitude of lips acceleration xyz
            7: eor_l        — left eye openness proxy (1 - normalized |vel_y|)

        Args:
            vid_folder  : path to raw session folder (where CSVs are saved)
            total_frames: number of frames after GT filtering
            num_clips   : number of chunks produced by preprocessing
            chunk_length: frames per chunk

        Returns:
            np.ndarray (num_clips, chunk_length, 8) or None if CSVs missing
        """
        base         = vid_folder.rstrip(os.sep)
        session_name = os.path.basename(base)
        subject_dir  = os.path.dirname(base)

        pose_path      = os.path.join(subject_dir, f"{session_name}_pose.csv")
        lips_path      = os.path.join(subject_dir, f"{session_name}_lips.csv")
        left_eye_path  = os.path.join(subject_dir, f"{session_name}_left_eye.csv")
        right_eye_path = os.path.join(subject_dir, f"{session_name}_right_eye.csv")

        required = [pose_path, lips_path, left_eye_path, right_eye_path]
        missing  = [p for p in required if not os.path.exists(p)]
        if missing:
            print(f"  [AyaLoader] Motion CSVs missing for {session_name}: "
                  f"{[os.path.basename(m) for m in missing]}")
            return None

        pose_df      = pd.read_csv(pose_path)
        lips_df      = pd.read_csv(lips_path)
        left_eye_df  = pd.read_csv(left_eye_path)
        right_eye_df = pd.read_csv(right_eye_path)

        n = min(len(pose_df), len(lips_df), len(left_eye_df),
                len(right_eye_df), total_frames)

        # --- Channels 3, 4, 5: yaw, pitch, roll ---
        yaw   = pose_df['yaw'].values[:n].astype(np.float32)
        pitch = pose_df['pitch'].values[:n].astype(np.float32)
        roll  = pose_df['roll'].values[:n].astype(np.float32)

        # --- Channel 2: lips_vel (L2 magnitude) ---
        lips_vel = np.sqrt(
            lips_df['vel_x'].values[:n]**2 +
            lips_df['vel_y'].values[:n]**2 +
            lips_df['vel_z'].values[:n]**2
        ).astype(np.float32)

        # --- Channel 6: lips_accel (L2 magnitude) ---
        lips_accel = np.sqrt(
            lips_df['acc_x'].values[:n]**2 +
            lips_df['acc_y'].values[:n]**2 +
            lips_df['acc_z'].values[:n]**2
        ).astype(np.float32)

        # --- Channels 0, 1: head_tx_vel, head_ty_vel ---
        # Left eye centroid moves rigidly with head — frame diff = translation vel
        eye_pos_x   = left_eye_df['pos_x'].values[:n].astype(np.float32)
        eye_pos_y   = left_eye_df['pos_y'].values[:n].astype(np.float32)
        head_tx_vel = np.gradient(eye_pos_x).astype(np.float32)
        head_ty_vel = np.gradient(eye_pos_y).astype(np.float32)

        # --- Channel 7: eor_l (eye openness ratio proxy) ---
        # When eye closes, vertical velocity of eye centroid increases.
        # Proxy: 1 - normalized |vel_y|, so open=1, closed=0.
        left_eye_vel_y = left_eye_df['vel_y'].values[:n].astype(np.float32)
        vel_range      = np.abs(left_eye_vel_y).max() + 1e-6
        eor_l          = (1.0 - np.abs(left_eye_vel_y) / vel_range).astype(np.float32)

        # --- Stack into (n, 8) ---
        motion_array = np.stack([
            head_tx_vel,   # 0
            head_ty_vel,   # 1
            lips_vel,      # 2
            yaw,           # 3
            pitch,         # 4
            roll,          # 5
            lips_accel,    # 6
            eor_l,         # 7
        ], axis=1)
        channel_names = ['head_tx_vel', 'head_ty_vel', 'lips_vel', 'yaw', 
                 'pitch', 'roll', 'lips_accel', 'eor_l']
        print(f"  [Motion] {session_name} — per-channel stats (mean, std, min, max):")
        for ci, name in enumerate(channel_names):
            ch = motion_array[:, ci]
            print(f"    {name:15s}: mean={ch.mean():8.3f} std={ch.std():8.3f} "
                f"min={ch.min():8.3f} max={ch.max():8.3f}")
        # Pad or trim to total_frames
        if motion_array.shape[0] < total_frames:
            pad = np.zeros((total_frames - motion_array.shape[0], MOTION_N_CHANNELS),
                           dtype=np.float32)
            motion_array = np.concatenate([motion_array, pad], axis=0)
        else:
            motion_array = motion_array[:total_frames]

        # Replace any NaNs/Infs with 0
        motion_array = np.nan_to_num(motion_array, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Slice into per-clip arrays ---
        motion_clips = []
        for clip_idx in range(num_clips):
            start = clip_idx * chunk_length
            end   = start + chunk_length
            clip  = motion_array[start:end]
            if clip.shape[0] < chunk_length:
                pad  = np.zeros((chunk_length - clip.shape[0], MOTION_N_CHANNELS),
                                dtype=np.float32)
                clip = np.concatenate([clip, pad], axis=0)
            motion_clips.append(clip)

        return np.stack(motion_clips, axis=0)  # (num_clips, chunk_length, 8)

    def _save_motion_clips(self, motion_clips, saved_filename):
        """Save each motion clip as a .npy file in the preprocessed cache folder.

        Naming: <saved_filename>_motion<clip_idx>.npy
        e.g.  : 3_motion0.npy, 3_motion1.npy, ...
        These are loaded automatically by BaseLoader.__getitem__.
        """
        for clip_idx, clip in enumerate(motion_clips):
            save_path = os.path.join(
                self.cached_path, f"{saved_filename}_motion{clip_idx}.npy")
            np.save(save_path, clip)

    # =========================================================================
    # PREPROCESSED DATA LOADING
    # =========================================================================

    def load_preprocessed_data(self):
        """Override: load preprocessed data AND motion metadata."""
        super().load_preprocessed_data()

        self.chunk_motion_map = {}
        meta_files = glob.glob(os.path.join(self.cached_path, "*_motion_meta.json"))
        for mf in meta_files:
            with open(mf, 'r') as f:
                self.chunk_motion_map.update(json.load(f))

        if self.chunk_motion_map:
            motion_counts = {}
            for ml in self.chunk_motion_map.values():
                motion_counts[ml] = motion_counts.get(ml, 0) + 1
            print(f"[AyaLoader] Motion metadata loaded: {motion_counts}")

    def get_motion_level(self, index):
        """Get motion level string for a given dataset index."""
        item_path = self.inputs[index]
        chunk_basename = os.path.basename(item_path)
        return self.chunk_motion_map.get(chunk_basename, "Unknown")

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @staticmethod
    def read_video(video_file):
        """Reads all .png frames from a directory, returns (T, H, W, 3)."""
        frames = []
        all_png = sorted(glob.glob(video_file + '*.png'))
        for png_path in all_png:
            img = cv2.imread(png_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        return np.asarray(frames)

    def read_video_list(self, png_list, resize_w=640, resize_h=360):
        """Read selected frames with immediate resize to save memory."""
        frames = []
        for png_path in png_list:
            img = cv2.imread(png_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
            frames.append(img)
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads BVP signal from pulse_data.csv."""
        df = pd.read_csv(bvp_file)
        return df["Signal_Value"].values.astype(np.float64)

'''"""The dataloader for Aya's custom rPPG dataset.

=== Key Modifications (v2) ===
1. SUBJECT-LEVEL SPLIT: Uses explicit subject ID lists instead of BEGIN/END ratio.
   Controlled via YAML: TRAIN_SUBJECTS, TEST_SUBJECTS
2. CONDITION FILTERING: Filters sessions by distance, motion, lux via config lists.
   Controlled in this file: TRAIN_CONDITIONS / TEST_CONDITIONS dicts.
3. MOTION-LEVEL METADATA: Each preprocessed chunk stores its motion_level
   (e.g., "Stationary", "Talking") in a side-car .npy file for granular evaluation.

Dataset Structure:
    DataRoot/
    |-- 0/                              <-- Subject folder (0~10)
    |   |-- vid_1m_30lux_Stationary_45deg_useiPhone/
    |   |   |-- 1709812345000.png       <-- Frame (filename = timestamp ms)
    |   |-- gt_1m_30lux_Stationary_45deg_useiPhone/
    |   |   |-- pulse_data.csv          <-- Columns: PC_Timestamp_ms, Signal_Value
    |-- 1/
    |-- ...
    |-- 10/
"""
import glob
import json
import os

import cv2
import numpy as np
import pandas as pd
from dataset.data_loader.BaseLoader import BaseLoader

# ============================================================================
#  CONFIGURATION: Edit these to control which sessions are loaded
# ============================================================================

# --- Subject Split ---
TRAIN_SUBJECTS = [1, 2, 3, 6, 8, 9, 10]
TEST_SUBJECTS  = [4, 5, 7]

# --- Train Conditions: which (distance, motion, lux) combos to include for training ---
# Set a list to filter, or None / empty list to include ALL values for that dimension.
TRAIN_CONDITIONS = {
    "distances": [1, 2, 3],               # e.g., [1, 2] for 1m and 2m
    "motions":   ['Nodding'],         # None = all motions (Stationary, Talking, etc.)
    "lux":       [70],                  # None = all lux levels
}

# --- Test Conditions: which combos to include for testing ---
# Typically broader to evaluate across all conditions
TEST_CONDITIONS = {
    "distances": [1, 2, 3],
    "motions":   ['Nodding'],                  # None = all motions
    "lux":       [70],
}


def _parse_folder_name(folder_name):
    """Parse a vid/gt folder name into its condition components.

    Example: 'vid_1m_30lux_Stationary_45deg_useiPhone'
    Returns: {'distance': 1, 'lux': 30, 'motion': 'Stationary', 'angle': '45deg', 'device': 'useiPhone'}
    """
    # Remove vid_ or gt_ prefix
    if folder_name.startswith("vid_"):
        core = folder_name[4:]
    elif folder_name.startswith("gt_"):
        core = folder_name[3:]
    else:
        core = folder_name

    parts = core.split("_")
    result = {}
    try:
        result["distance"] = int(parts[0].replace("m", ""))   # "1m" -> 1
        result["lux"] = int(parts[1].replace("lux", ""))       # "30lux" -> 30
        result["motion"] = parts[2]                             # "Stationary"
        result["angle"] = parts[3] if len(parts) > 3 else ""   # "45deg"
        result["device"] = parts[4] if len(parts) > 4 else ""  # "useiPhone"
    except (IndexError, ValueError):
        result["distance"] = -1
        result["lux"] = -1
        result["motion"] = "Unknown"
        result["angle"] = ""
        result["device"] = ""
    return result


def _matches_conditions(parsed, conditions):
    """Check if a parsed folder matches the given condition filters.

    Args:
        parsed: dict from _parse_folder_name
        conditions: dict with keys 'distances', 'motions', 'lux' (each a list or None)
    Returns:
        True if the folder matches ALL non-None filters.
    """
    if conditions.get("distances"):
        if parsed["distance"] not in conditions["distances"]:
            return False
    if conditions.get("motions"):
        if parsed["motion"] not in conditions["motions"]:
            return False
    if conditions.get("lux"):
        if parsed["lux"] not in conditions["lux"]:
            return False
    return True


class AyaLoader(BaseLoader):
    """The data loader for Aya's custom rPPG dataset (v2 with subject split + condition filtering)."""

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes the Aya dataloader.

        Args:
            name(str): "train", "valid", or "test" — determines which subject/condition list to use.
            data_path(str): path to the root folder containing subject directories (0~10).
            config_data(CfgNode): data settings (ref: config.py).
        """
        # Store split name BEFORE calling super().__init__ which triggers preprocessing
        self.split_name = name  # "train", "valid", or "test"

        # Metadata: maps chunk filename (str) -> motion_level (str)
        # Populated during preprocessing, saved/loaded as JSON sidecar
        self.chunk_motion_map = {}

        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Scan dataset root and return all valid vid/gt pairs with metadata.

        Filters by SUBJECT list and CONDITION list based on self.split_name.
        """
        # Select subjects and conditions based on split
        if self.split_name in ("train",):
            allowed_subjects = TRAIN_SUBJECTS
            conditions = TRAIN_CONDITIONS
        else:  # "valid" or "test"
            allowed_subjects = TEST_SUBJECTS
            conditions = TEST_CONDITIONS

        data_dirs = []
        subject_folders = sorted(
            [d for d in os.listdir(data_path)
             if os.path.isdir(os.path.join(data_path, d)) and d.isdigit()],
            key=lambda x: int(x)
        )
        if not subject_folders:
            raise ValueError(self.dataset_name + " data paths empty!")

        trial_counter = 0
        for subj_folder in subject_folders:
            subject = int(subj_folder)

            # --- Subject filter ---
            if subject not in allowed_subjects:
                continue

            subj_path = os.path.join(data_path, subj_folder)
            vid_folders = sorted(glob.glob(os.path.join(subj_path, "vid_*")))

            for vid_folder in vid_folders:
                vid_name = os.path.basename(vid_folder)
                parsed = _parse_folder_name(vid_name)

                # --- Condition filter ---
                if not _matches_conditions(parsed, conditions):
                    continue

                gt_name = "gt_" + vid_name[4:]
                gt_folder = os.path.join(subj_path, gt_name)

                if os.path.isdir(gt_folder):
                    data_dirs.append({
                        "index": trial_counter,
                        "path": vid_folder,
                        "subject": subject,
                        "gt_path": gt_folder,
                        "motion_level": parsed["motion"],   # <-- NEW: store motion
                        "distance": parsed["distance"],
                        "lux": parsed["lux"],
                    })
                    trial_counter += 1

        if not data_dirs:
            raise ValueError(
                f"{self.dataset_name} [{self.split_name}] no valid vid/gt pairs found! "
                f"Subjects={allowed_subjects}, Conditions={conditions}"
            )

        print(f"\n[AyaLoader] Split='{self.split_name}', "
              f"Subjects={allowed_subjects}, "
              f"Sessions found={len(data_dirs)}")
        for d in data_dirs:
            print(f"  idx={d['index']} subj={d['subject']} "
                  f"motion={d['motion_level']} dist={d['distance']}m "
                  f"lux={d['lux']} -> {os.path.basename(d['path'])}")
        print()

        return data_dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Override: subject filtering is already done in get_raw_data.

        BEGIN/END are ignored — the split is fully controlled by subject lists.
        Return all data_dirs as-is.
        """
        return data_dirs

    def multi_process_manager(self, data_dirs, config_preprocess, multi_process_quota=2):
        """Override: limit parallel processes."""
        return super().multi_process_manager(data_dirs, config_preprocess, multi_process_quota=2)

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Preprocess one session: read frames, align GT, crop face, chunk, save.

        Also saves motion_level metadata for each chunk.
        """
        saved_filename = data_dirs[i]['index']
        vid_path = data_dirs[i]['path']
        gt_path = data_dirs[i]['gt_path']
        motion_level = data_dirs[i]['motion_level']
        print(f"[{i}] START", flush=True)
        print(f"\n[{i + 1}/{len(data_dirs)}] Processing: {os.path.basename(vid_path)} "
              f"(motion={motion_level})")

        # ---- Step 1: Read & trim ground truth ----
        gt_csv = os.path.join(gt_path, "pulse_data.csv")
        if not os.path.exists(gt_csv):
            print(f"[WARN] GT not found, skipping: {gt_csv}")
            file_list_dict[i] = []
            return
        print(f"[{i}] GT found", flush=True)

        gt_df = pd.read_csv(gt_csv)
        print(f"[{i}] GT loaded, rows={len(gt_df)}", flush=True)
        gt_df = gt_df[["PC_Timestamp_ms", "Signal_Value"]].dropna()
        trim = max(10, int(len(gt_df) * 0.025))  # 2.5% each end, minimum 10 samples
        gt_df = gt_df.iloc[trim:-trim].reset_index(drop=True)
        print(f"[{i}] GT trimmed, rows={len(gt_df)}", flush=True)

        if len(gt_df) < 30:
            print(f"[WARN] GT too short after trim, skipping: {gt_csv}")
            file_list_dict[i] = []
            return

        gt_timestamps = gt_df["PC_Timestamp_ms"].values.astype(np.float64)
        gt_signal = gt_df["Signal_Value"].values.astype(np.float64)

        # ---- Step 2: Get filenames & timestamps (no pixel loading yet) ----
        all_png = sorted(glob.glob(vid_path + os.sep + '*.png'))
        frame_ts = np.array([
            float(os.path.splitext(os.path.basename(p))[0]) for p in all_png
        ])

        # ---- Step 3: Filter to GT time range ----
        gt_start, gt_end = gt_timestamps[0], gt_timestamps[-1]
        mask = (frame_ts >= gt_start) & (frame_ts <= gt_end)
        selected_png = [p for p, m in zip(all_png, mask) if m]
        frame_ts = frame_ts[mask]

        if len(selected_png) < 30:
            print(f"[WARN] Too few frames after alignment, skipping: {vid_path}")
            file_list_dict[i] = []
            return

        # ---- Step 4: Align GT signal to frame timestamps ----
        indices = np.searchsorted(gt_timestamps, frame_ts, side='left')
        indices = np.clip(indices, 0, len(gt_timestamps) - 1)
        # nearest-neighbor: compare left and right candidate
        indices_left = np.clip(indices - 1, 0, len(gt_timestamps) - 1)
        left_dist = np.abs(gt_timestamps[indices_left] - frame_ts)
        right_dist = np.abs(gt_timestamps[indices] - frame_ts)
        indices = np.where(left_dist < right_dist, indices_left, indices)
        bvps = gt_signal[indices]

        print(f"[{i}] Reading frames...", flush=True)

        # ---- Step 5: Read frames (memory efficient pre-resize) ----
        frames = self.read_video_list(selected_png, resize_w=640, resize_h=360)

        print(f"[{i}] Frames read", flush=True)
        # ---- Step 6: Preprocess (face crop, normalize, chunk) & Save ----
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        print(f"[{i}] Preprocessing...", flush=True)
        input_name_list, label_name_list = self.save_multi_process(
            frames_clips, bvps_clips, saved_filename)

        # ---- Step 7: Save motion_level metadata for each chunk ----
        # Maps each chunk's input filename -> motion level
        print(f"[{i}] Saving...", flush=True)
        motion_map = {}
        for inp_path in input_name_list:
            chunk_basename = os.path.basename(inp_path)  # e.g., "3_input0.npy"
            motion_map[chunk_basename] = motion_level

        # Save as JSON sidecar next to the cached data
        meta_path = os.path.join(
            self.cached_path,
            f"{saved_filename}_motion_meta.json"
        )
        with open(meta_path, 'w') as f:
            json.dump(motion_map, f)

        file_list_dict[i] = input_name_list

    def load_preprocessed_data(self):
        """Override: load preprocessed data AND motion metadata."""
        super().load_preprocessed_data()

        # Load all motion metadata JSONs
        self.chunk_motion_map = {}
        meta_files = glob.glob(os.path.join(self.cached_path, "*_motion_meta.json"))
        for mf in meta_files:
            with open(mf, 'r') as f:
                self.chunk_motion_map.update(json.load(f))

        if self.chunk_motion_map:
            # Count per motion level
            motion_counts = {}
            for ml in self.chunk_motion_map.values():
                motion_counts[ml] = motion_counts.get(ml, 0) + 1
            print(f"[AyaLoader] Motion metadata loaded: {motion_counts}")

    def get_motion_level(self, index):
        """Get motion level for a given dataset index.

        Useful for granular test evaluation.
        Returns: str like 'Stationary', 'Talking', etc. or 'Unknown'
        """
        item_path = self.inputs[index]
        chunk_basename = os.path.basename(item_path)
        return self.chunk_motion_map.get(chunk_basename, "Unknown")

    # ---- Utility methods (unchanged) ----

    @staticmethod
    def read_video(video_file):
        """Reads all .png frames from a directory, returns (T, H, W, 3)."""
        frames = []
        all_png = sorted(glob.glob(video_file + '*.png'))
        for png_path in all_png:
            img = cv2.imread(png_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        return np.asarray(frames)

    def read_video_list(self, png_list, resize_w=640, resize_h=360):
        """Read selected frames with immediate resize to save memory."""
        frames = []
        for png_path in png_list:
            img = cv2.imread(png_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
            frames.append(img)
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads BVP signal from pulse_data.csv."""
        df = pd.read_csv(bvp_file)
        return df["Signal_Value"].values.astype(np.float64) '''