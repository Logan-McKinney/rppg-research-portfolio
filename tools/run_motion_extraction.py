"""
run_motion_extraction.py
========================
Batch runner that calls Patrick's MediaPipe motion tracker on every
PURE subject-trial folder and saves one CSV per subject-trial.

Folder structure assumed:
    <local-path>
        01-01\\
            01-01\\          <- raw PNG frames live here
            01-01.json
        01-02\\
            01-02\\
            01-02.json
        ...
        10-06\\
            10-06\\
            10-06.json

Output CSVs are saved to:
    <local-path>
        01-01_motion.csv
        01-02_motion.csv
        ...
        10-06_motion.csv

HOW TO RUN:
    conda activate rppg-toolbox
    cd <local-path>
    python tools/run_motion_extraction.py

The face_landmarker.task model file must be present in the working
directory (same folder you run the script from). Download it from:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""

import os
import glob
import re
import csv
import numpy as np
from collections import deque

import cv2
import mediapipe as mp

# ─────────────────────────────────────────────
#  PATHS — edit if your layout differs
# ─────────────────────────────────────────────
PURE_ROOT    = r"data/PURE"
OUTPUT_DIR   = r"data/motion/PURE"
FPS          = 30.0
MODEL_PATH   = "face_landmarker.task"   # must be in working directory

# ─────────────────────────────────────────────
#  Motion tracker configuration
#  (matches Patrick's script exactly)
# ─────────────────────────────────────────────
LANDMARK_GROUPS = {
    'lips':      [13, 14, 78, 308],
    'l_eyebrow': [46,  53,  52,  65],
    'r_eyebrow': [285, 295, 282, 283],
    'l_eye':     [33,  133],
    'r_eye':     [362, 263],
    'l_cheek':   [234, 50,  205, 187],
    'r_cheek':   [454, 280, 425, 411],
}

EYE_OPEN_IDX = {
    'l_eye': {'top': 159, 'bot': 145, 'inner': 133, 'outer':  33},
    'r_eye': {'top': 386, 'bot': 374, 'inner': 263, 'outer': 362},
}

NOSE_REF_IDX   = [1, 4, 5, 6, 168, 197]
SMOOTH_WINDOW  = 8
MOVE_THRESHOLD = 8

CSV_FIELDNAMES = [
    'frame_index', 'filename', 'timestamp_sec',
    'pitch', 'roll', 'yaw', 'dist_cm',
    'lips_vel', 'lips_accel',
    'l_eyebrow_vel', 'l_eyebrow_accel',
    'r_eyebrow_vel', 'r_eyebrow_accel',
    'l_eye_vel', 'l_eye_accel',
    'r_eye_vel', 'r_eye_accel',
    'l_cheek_vel', 'l_cheek_accel',
    'r_cheek_vel', 'r_cheek_accel',
    'eor_l', 'eor_r',
    'head_tx_vel', 'head_ty_vel',
    'face_detected',
]

# ─────────────────────────────────────────────
#  Helpers (from Patrick's script)
# ─────────────────────────────────────────────
def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def get_avg_pos(landmarks, indices, w, h):
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
    return pts.mean(axis=0)

def rotation_to_euler(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        roll  = np.degrees(np.arctan2( R[2, 1], R[2, 2]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = np.degrees(np.arctan2( R[1, 0], R[0, 0]))
    else:
        roll  = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw   = 0.0
    return pitch, roll, yaw

def eye_openness_ratio(landmarks, idx_dict, w, h):
    top   = np.array([landmarks[idx_dict['top']].x * w,  landmarks[idx_dict['top']].y * h])
    bot   = np.array([landmarks[idx_dict['bot']].x * w,  landmarks[idx_dict['bot']].y * h])
    inner = np.array([landmarks[idx_dict['inner']].x * w, landmarks[idx_dict['inner']].y * h])
    outer = np.array([landmarks[idx_dict['outer']].x * w, landmarks[idx_dict['outer']].y * h])
    vertical   = np.linalg.norm(top - bot)
    horizontal = np.linalg.norm(inner - outer)
    return vertical / horizontal if horizontal > 0 else 0.0

# ─────────────────────────────────────────────
#  Core: process one subject-trial folder
# ─────────────────────────────────────────────
def process_subject(subject_name, img_dir, csv_path, detector):
    """
    Run motion extraction on all PNG frames in img_dir.
    Saves results to csv_path.
    subject_name: e.g. '01-01'
    img_dir:      e.g. 'data/LOCAL_INPUT'  (folder with raw PNGs)
    csv_path:     e.g. 'data/motion/PURE'
    """
    all_files = sorted(
        [f for f in os.listdir(img_dir) if f.lower().endswith('.png')],
        key=natural_sort_key
    )
    total_frames = len(all_files)
    if total_frames == 0:
        print(f"  WARNING: No PNG files found in {img_dir}, skipping.")
        return

    print(f"  {subject_name}: {total_frames} frames -> {csv_path}")

    dt = 1.0 / FPS

    vel_history = {name: deque(maxlen=SMOOTH_WINDOW) for name in LANDMARK_GROUPS}
    eor_history = {name: deque(maxlen=SMOOTH_WINDOW) for name in EYE_OPEN_IDX}
    pos_history = deque(maxlen=5)

    vel_data  = {name: (0.0, 0.0) for name in LANDMARK_GROUPS}
    eor       = {'l_eye': 0.0, 'r_eye': 0.0}
    pitch = roll = yaw = dist_cm = 0.0
    head_tx = head_ty = 0.0
    prev_nose = None

    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for frame_idx, filename in enumerate(all_files):
            img_path      = os.path.join(img_dir, filename)
            frame         = cv2.imread(img_path)
            if frame is None:
                print(f"    WARNING: could not read {img_path}, skipping frame.")
                continue

            h, w, _       = frame.shape
            timestamp_sec = frame_idx / FPS
            timestamp_ms  = int(timestamp_sec * 1000)

            if frame_idx % 100 == 0:
                print(f"    Frame {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.1f}%)")

            # Run MediaPipe
            mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result       = detector.detect_for_video(image=mp_image, timestamp_ms=timestamp_ms)
            face_detected = bool(result.face_landmarks and result.facial_transformation_matrixes)

            if face_detected:
                face_lm = result.face_landmarks[0]

                # Head pose
                matrix   = np.array(result.facial_transformation_matrixes[0].data).reshape(4, 4)
                dist_cm  = abs(matrix[2, 3])
                yaw, roll, pitch = rotation_to_euler(matrix[:3, :3])

                # Nose reference
                nose_pos = get_avg_pos(face_lm, NOSE_REF_IDX, w, h)
                if prev_nose is not None:
                    nose_vel_vec = (nose_pos - prev_nose) / dt
                    head_tx      = float(nose_vel_vec[0])
                    head_ty      = float(nose_vel_vec[1])
                else:
                    nose_vel_vec = np.zeros(2)
                    head_tx = head_ty = 0.0
                prev_nose = nose_pos

                # Positions per landmark group
                current_positions = {
                    name: get_avg_pos(face_lm, indices, w, h)
                    for name, indices in LANDMARK_GROUPS.items()
                }
                current_positions['_nose_ref'] = nose_pos

                # Eye openness
                for eye_name, idx_dict in EYE_OPEN_IDX.items():
                    raw_eor = eye_openness_ratio(face_lm, idx_dict, w, h)
                    eor_history[eye_name].append(raw_eor)
                    eor[eye_name] = float(np.mean(eor_history[eye_name]))

                # Velocity and acceleration per group
                if len(pos_history) > 0:
                    for name in LANDMARK_GROUPS:
                        cur_pos  = current_positions[name]
                        prev_pos = pos_history[-1][name]
                        raw_vel_vec = (cur_pos - prev_pos) / dt
                        rel_vel_vec = raw_vel_vec - nose_vel_vec
                        rel_speed   = float(np.linalg.norm(rel_vel_vec))

                        vel_history[name].append(rel_speed)
                        smooth_vel = float(np.mean(vel_history[name]))

                        if len(vel_history[name]) >= 2:
                            accel = (vel_history[name][-1] - vel_history[name][-2]) / dt
                        else:
                            accel = 0.0

                        if name in eor_history and len(eor_history[name]) >= 2:
                            eor_delta        = abs(eor_history[name][-1] - eor_history[name][-2])
                            eye_blink_signal = eor_delta * 40 / dt
                            smooth_vel       = max(smooth_vel, eye_blink_signal)

                        vel_data[name] = (smooth_vel, accel)

                pos_history.append(current_positions)

            # Write row — zeros for all features if no face detected
            row = {
                'frame_index':     frame_idx,
                'filename':        filename,
                'timestamp_sec':   round(timestamp_sec, 4),
                'pitch':           round(pitch, 3),
                'roll':            round(roll, 3),
                'yaw':             round(yaw, 3),
                'dist_cm':         round(dist_cm, 2),
                'lips_vel':        round(vel_data['lips'][0], 3),
                'lips_accel':      round(vel_data['lips'][1], 3),
                'l_eyebrow_vel':   round(vel_data['l_eyebrow'][0], 3),
                'l_eyebrow_accel': round(vel_data['l_eyebrow'][1], 3),
                'r_eyebrow_vel':   round(vel_data['r_eyebrow'][0], 3),
                'r_eyebrow_accel': round(vel_data['r_eyebrow'][1], 3),
                'l_eye_vel':       round(vel_data['l_eye'][0], 3),
                'l_eye_accel':     round(vel_data['l_eye'][1], 3),
                'r_eye_vel':       round(vel_data['r_eye'][0], 3),
                'r_eye_accel':     round(vel_data['r_eye'][1], 3),
                'l_cheek_vel':     round(vel_data['l_cheek'][0], 3),
                'l_cheek_accel':   round(vel_data['l_cheek'][1], 3),
                'r_cheek_vel':     round(vel_data['r_cheek'][0], 3),
                'r_cheek_accel':   round(vel_data['r_cheek'][1], 3),
                'eor_l':           round(eor['l_eye'], 4),
                'eor_r':           round(eor['r_eye'], 4),
                'head_tx_vel':     round(head_tx, 3),
                'head_ty_vel':     round(head_ty, 3),
                'face_detected':   int(face_detected),
            }
            writer.writerow(row)

    print(f"  Done: {subject_name}")

# ─────────────────────────────────────────────
#  Main — iterate over all PURE subject folders
# ─────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all subject-trial folders matching the PURE naming pattern XX-XX
    subject_dirs = sorted(
        [d for d in glob.glob(os.path.join(PURE_ROOT, "*-*")) if os.path.isdir(d)],
        key=lambda p: natural_sort_key(os.path.basename(p))
    )

    if not subject_dirs:
        raise RuntimeError(f"No subject folders found in {PURE_ROOT}")

    print(f"Found {len(subject_dirs)} subject-trial folders.")
    print(f"Output CSVs will be saved to: {OUTPUT_DIR}\n")

    # Set up MediaPipe detector once — reused across all subjects
    # Using VIDEO mode so timestamps are used for tracking within each subject.
    # Note: detector is recreated per subject below to reset tracking state.
    BaseOptions       = mp.tasks.BaseOptions
    FaceLandmarker    = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )

    for subject_dir in subject_dirs:
        subject_name = os.path.basename(subject_dir)  # e.g. '01-01'

        # Raw PNG frames are in a subfolder with the same name as the subject
        # e.g. PURE\01-01\01-01\Image*.png
        img_dir = os.path.join(subject_dir, subject_name)

        if not os.path.isdir(img_dir):
            print(f"WARNING: Expected image folder not found: {img_dir}, skipping.")
            continue

        # Output CSV path — named consistently to match subject folder
        csv_path = os.path.join(OUTPUT_DIR, f"{subject_name}_motion.csv")

        # Skip if already done — allows resuming if interrupted
        if os.path.exists(csv_path):
            print(f"  Skipping {subject_name} — CSV already exists.")
            continue

        # Recreate detector per subject to reset VIDEO mode tracking state.
        # This prevents timestamps from a previous subject leaking into the next.
        detector = FaceLandmarker.create_from_options(options)

        print(f"Processing {subject_name}...")
        process_subject(subject_name, img_dir, csv_path, detector)
        detector.close()

    print("\nAll subjects complete.")
    print(f"CSVs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()