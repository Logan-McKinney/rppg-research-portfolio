"""The dataloader for PURE datasets.

Details for the PURE Dataset see https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure
If you use this dataset, please cite the following publication:
Stricker, R., Müller, S., Gross, H.-M.
Non-contact Video-based Pulse Rate Measurement on a Mobile Service Robot
in: Proc. 23st IEEE Int. Symposium on Robot and Human Interactive Communication (Ro-Man 2014), Edinburgh, Scotland, UK, pp. 1056 - 1062, IEEE 2014
"""
import glob
import json
import os
import re

import cv2
import numpy as np
import pandas as pd  # ADDED: needed to load motion CSVs
from .BaseLoader import BaseLoader
from tqdm import tqdm


# ADDED: columns from Patrick's CSV that get fed into the model's noise branch.
# Order matters — must match the 8-channel layout expected by DeepPhysMultiTask_v1:
#   [head_tx_vel, head_ty_vel, lips_vel, yaw, pitch, roll, lips_accel, eor_l]
MOTION_COLUMNS = [
    'head_tx_vel',   # channel 0: head translation velocity x
    'head_ty_vel',   # channel 1: head translation velocity y
    'lips_vel',      # channel 2: lip landmark velocity
    'yaw',           # channel 3: head yaw
    'pitch',         # channel 4: head pitch
    'roll',          # channel 5: head roll
    'lips_accel',    # channel 6: lip acceleration
    'eor_l',         # channel 7: left eye openness ratio
]


class PURELoader(BaseLoader):
    """The data loader for the PURE dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an PURE dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- 01-01/
                     |      |-- 01-01/
                     |      |-- 01-01.json
                     |   |-- 01-02/
                     |      |-- 01-02/
                     |      |-- 01-02.json
                     |...
                     |   |-- ii-jj/
                     |      |-- ii-jj/
                     |      |-- ii-jj.json
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For PURE dataset)."""
        data_dirs = glob.glob(data_path + os.sep + "*-*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        for data_dir in data_dirs:
            subject_trail_val = os.path.split(data_dir)[-1].replace('-', '')
            index = int(subject_trail_val)
            subject = int(subject_trail_val[0:2])
            dirs.append({"index": index, "path": data_dir, "subject": subject})
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values,
        and ensures no overlapping subjects between splits"""
        if begin == 0 and end == 1:
            return data_dirs

        data_info = dict()
        for data in data_dirs:
            subject = data['subject']
            data_dir = data['path']
            index = data['index']
            if subject not in data_info:
                data_info[subject] = []
            data_info[subject].append({"index": index, "path": data_dir, "subject": subject})

        subj_list = sorted(list(data_info.keys()))
        num_subjs = len(subj_list)
        subj_range = list(range(0, num_subjs))
        if begin != 0 or end != 1:
            subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))

        data_dirs_new = []
        for i in subj_range:
            subj_num = subj_list[i]
            subj_files = data_info[subj_num]
            data_dirs_new += subj_files
        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Invoked by preprocess_dataset for multi_process."""
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        frames = self.read_video(
            os.path.join(data_dirs[i]['path'], filename, ""))
        bvps = self.read_wave(
            os.path.join(data_dirs[i]['path'], "{0}.json".format(filename)))
        target_length = frames.shape[0]
        bvps = BaseLoader.resample_ppg(bvps, target_length)
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)

        # ADDED: load motion CSV for this subject and slice into per-clip arrays.
        # The CSV lives at PURE_MotionFeatures/<subject_name>_motion.csv
        # e.g. <local-path>
        motion_clips = self._load_motion_clips(
            subject_name=filename,
            total_frames=target_length,
            num_clips=frames_clips.shape[0],
            chunk_length=config_preprocess.CHUNK_LENGTH,
        )

        # Save input and label clips (unchanged from original)
        input_name_list, label_name_list = self.save_multi_process(
            frames_clips, bvps_clips, saved_filename
        )

        # ADDED: save motion clips as separate .npy files alongside input/label files.
        # Naming convention: <saved_filename>_motion<clip_index>.npy
        # e.g. 101_motion0.npy, 101_motion1.npy, etc.
        # These live in the same CACHED_PATH folder as the input/label .npy files.
        if motion_clips is not None:
            self._save_motion_clips(motion_clips, saved_filename)

        file_list_dict[i] = input_name_list

    # ADDED: loads the motion CSV for a subject, slices it into clip-length chunks,
    # and returns an array of shape (num_clips, chunk_length, 8).
    # Returns None if the CSV does not exist so the rest of preprocessing still works.
    def _load_motion_clips(self, subject_name, total_frames, num_clips, chunk_length):
        """
        Load motion features from Patrick's CSV and slice into per-clip arrays.

        Args:
            subject_name (str): e.g. '01-01' — used to find the CSV file.
            total_frames (int): total number of frames in the raw video.
            num_clips (int): number of clips produced by chunking.
            chunk_length (int): number of frames per clip (e.g. 180).

        Returns:
            np.ndarray of shape (num_clips, chunk_length, 8), or None if CSV missing.
        """
        # CSV path: <local-path>
        csv_path = os.path.join(
            r"data/motion/PURE",
            f"{subject_name}_motion.csv"
        )

        if not os.path.exists(csv_path):
            print(f"  WARNING: Motion CSV not found for {subject_name} at {csv_path}. "
                  f"Motion features will not be saved for this subject.")
            return None

        df = pd.read_csv(csv_path)

        # Fill any NaN values with 0 — can happen on frames where face was not detected
        df[MOTION_COLUMNS] = df[MOTION_COLUMNS].fillna(0.0)

        # Extract only the 8 columns the model uses, as a numpy array (total_frames, 8)
        motion_array = df[MOTION_COLUMNS].values.astype(np.float32)

        # Pad or trim to match total_frames in case of minor frame count mismatch
        if motion_array.shape[0] < total_frames:
            pad = np.zeros((total_frames - motion_array.shape[0], 8), dtype=np.float32)
            motion_array = np.concatenate([motion_array, pad], axis=0)
        elif motion_array.shape[0] > total_frames:
            motion_array = motion_array[:total_frames]

        # Slice into clips — clip i covers frames [i*chunk_length : (i+1)*chunk_length]
        # This matches exactly how frames_clips are chunked in BaseLoader.chunk()
        motion_clips = []
        for clip_idx in range(num_clips):
            start = clip_idx * chunk_length
            end   = start + chunk_length
            clip  = motion_array[start:end]

            # If last clip is short (edge case), pad with zeros
            if clip.shape[0] < chunk_length:
                pad  = np.zeros((chunk_length - clip.shape[0], 8), dtype=np.float32)
                clip = np.concatenate([clip, pad], axis=0)

            motion_clips.append(clip)

        return np.stack(motion_clips, axis=0)  # (num_clips, chunk_length, 8)

    # ADDED: saves each motion clip as a .npy file in the same CACHED_PATH folder
    # as the input/label files so everything stays together.
    def _save_motion_clips(self, motion_clips, saved_filename):
        """
        Save motion clips as individual .npy files.

        Args:
            motion_clips: np.ndarray of shape (num_clips, chunk_length, 8)
            saved_filename: int or str used as the base name, e.g. 101

        Saves files like:
            <CACHED_PATH>/<cache_folder>/101_motion0.npy
            <CACHED_PATH>/<cache_folder>/101_motion1.npy
            ...
        """
        for clip_idx, clip in enumerate(motion_clips):
            filename = f"{saved_filename}_motion{clip_idx}.npy"
            save_path = os.path.join(self.cached_path, filename)
            np.save(save_path, clip)

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3)"""
        frames = list()
        all_png = sorted(glob.glob(video_file + '*.png'))
        for png_path in all_png:
            img = cv2.imread(png_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            labels = json.load(f)
            waves = [label["Value"]["waveform"]
                     for label in labels["/FullPackage"]]
        return np.asarray(waves)

#---------------------------------------------------OLD----------------------------------------------------------
'''"""The dataloader for PURE datasets.

Details for the PURE Dataset see https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure
If you use this dataset, please cite the following publication:
Stricker, R., Müller, S., Gross, H.-M.
Non-contact Video-based Pulse Rate Measurement on a Mobile Service Robot
in: Proc. 23st IEEE Int. Symposium on Robot and Human Interactive Communication (Ro-Man 2014), Edinburgh, Scotland, UK, pp. 1056 - 1062, IEEE 2014
"""
import glob
import glob
import json
import os
import re

import cv2
import numpy as np
# from dataset.data_loader.BaseLoader import BaseLoader
from .BaseLoader import BaseLoader
from tqdm import tqdm


class PURELoader(BaseLoader):
    """The data loader for the PURE dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an PURE dataloader.
            Args:
                data_path(str): path of a folder which stores raw video and bvp data.
                e.g. data_path should be "RawData" for below dataset structure:
                -----------------
                     RawData/
                     |   |-- 01-01/
                     |      |-- 01-01/
                     |      |-- 01-01.json
                     |   |-- 01-02/
                     |      |-- 01-02/
                     |      |-- 01-02.json
                     |...
                     |   |-- ii-jj/
                     |      |-- ii-jj/
                     |      |-- ii-jj.json
                -----------------
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For PURE dataset)."""

        data_dirs = glob.glob(data_path + os.sep + "*-*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        for data_dir in data_dirs:
            subject_trail_val = os.path.split(data_dir)[-1].replace('-', '')
            index = int(subject_trail_val)
            subject = int(subject_trail_val[0:2])
            dirs.append({"index": index, "path": data_dir, "subject": subject})
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values,
        and ensures no overlapping subjects between splits"""

        # return the full directory
        if begin == 0 and end == 1:
            return data_dirs

        # get info about the dataset: subject list and num vids per subject
        data_info = dict()
        for data in data_dirs:
            subject = data['subject']
            data_dir = data['path']
            index = data['index']
            # creates a dictionary of data_dirs indexed by subject number
            if subject not in data_info:  # if subject not in the data info dictionary
                data_info[subject] = []  # make an emplty list for that subject
            # append a tuple of the filename, subject num, trial num, and chunk num
            data_info[subject].append({"index": index, "path": data_dir, "subject": subject})

        subj_list = list(data_info.keys())  # all subjects by number ID (1-27)
        subj_list = sorted(subj_list)
        num_subjs = len(subj_list)  # number of unique subjects

        # get split of data set (depending on start / end)
        subj_range = list(range(0, num_subjs))
        if begin != 0 or end != 1:
            subj_range = list(range(int(begin * num_subjs), int(end * num_subjs)))

        # compile file list
        data_dirs_new = []
        for i in subj_range:
            subj_num = subj_list[i]
            subj_files = data_info[subj_num]
            data_dirs_new += subj_files  # add file information to file_list (tuple of fname, subj ID, trial num,
            # chunk num)

        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ Invoked by preprocess_dataset for multi_process. """
        filename = os.path.split(data_dirs[i]['path'])[-1]
        saved_filename = data_dirs[i]['index']

        frames = self.read_video(
            os.path.join(data_dirs[i]['path'], filename, ""))
        bvps = self.read_wave(
            os.path.join(data_dirs[i]['path'], "{0}.json".format(filename)))
        target_length = frames.shape[0]
        bvps = BaseLoader.resample_ppg(bvps, target_length)
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        frames = list()
        all_png = sorted(glob.glob(video_file + '*.png'))
        for png_path in all_png:
            img = cv2.imread(png_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
        return np.asarray(frames)

    @staticmethod
    def read_wave(bvp_file):
        """Reads a bvp signal file."""
        with open(bvp_file, "r") as f:
            labels = json.load(f)
            waves = [label["Value"]["waveform"]
                     for label in labels["/FullPackage"]]
        return np.asarray(waves)
'''