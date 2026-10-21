from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# ----------------------------
# Config container (optional)
# ----------------------------
@dataclass
class ROIConfig:
    # Resize output size
    W: int
    H: int

    # Whether to crop a face region at all
    CROP_FACE: bool = True

    # Face detection options
    DYNAMIC_DETECTION: bool = True
    DYNAMIC_DETECTION_FREQUENCY: int = 30

    # Bounding-box options
    LARGE_FACE_BOX: bool = False
    LARGE_BOX_COEF: float = 1.0

    # Path to cascade xml (default matches your repo)
    CASCADE_PATH: str = "./dataset/haarcascade_frontalface_default.xml"


# ----------------------------
# Base interface
# ----------------------------
class ROIExtractorBase:
    """Base interface for ROI extraction.
    Input: frames (T, H0, W0, 3) RGB uint8/float
    Output: roi_frames (T, H, W, 3) + optional metadata
    """

    def extract(self, frames: np.ndarray) -> Tuple[np.ndarray, Dict]:
        raise NotImplementedError


# ----------------------------
# Haar Cascade ROI extractor
# ----------------------------
class HaarCascadeROIExtractor(ROIExtractorBase):
    def __init__(self, cfg: ROIConfig):
        self.cfg = cfg
        self._detector = None  # lazy init — cv2.CascadeClassifier cannot be pickled

    def _get_detector(self):
        if self._detector is None or self._detector.empty():
            self._detector = cv2.CascadeClassifier(self.cfg.CASCADE_PATH)
            if self._detector.empty():
                raise FileNotFoundError(
                    f"Could not load Haar cascade at '{self.cfg.CASCADE_PATH}'. "
                    f"Check path relative to your working directory."
                )
        return self._detector

    def _detect_face_box(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Returns face box as [x, y, w, h]. If none found, returns full frame box."""
        detector = self._get_detector()
        face_zone = detector.detectMultiScale(frame_bgr)

        if len(face_zone) < 1:
            h, w = frame_bgr.shape[:2]
            box = np.array([0, 0, w, h], dtype=np.int32)
            return box

        if len(face_zone) >= 2:
            areas = face_zone[:, 2] * face_zone[:, 3]
            box = face_zone[np.argmax(areas)]
        else:
            box = face_zone[0]

        x, y, bw, bh = box

        if self.cfg.LARGE_FACE_BOX:
            coef = float(self.cfg.LARGE_BOX_COEF)
            dx = int(((coef - 1.0) / 2.0) * bw)
            dy = int(((coef - 1.0) / 2.0) * bh)
            x = max(0, x - dx)
            y = max(0, y - dy)
            bw = int(coef * bw)
            bh = int(coef * bh)

        return np.array([x, y, bw, bh], dtype=np.int32)

    def extract(self, frames: np.ndarray) -> Tuple[np.ndarray, Dict]:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected frames shape (T,H,W,3). Got {frames.shape}.")

        T = frames.shape[0]
        out_h, out_w = int(self.cfg.H), int(self.cfg.W)

        def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
            return img_rgb[:, :, ::-1]

        if self.cfg.DYNAMIC_DETECTION:
            det_freq = max(1, int(self.cfg.DYNAMIC_DETECTION_FREQUENCY))
            num_det = ceil(T / det_freq)
        else:
            det_freq = T  # detect once at frame 0
            num_det = 1

        face_boxes = []
        for idx in range(num_det):
            frame_idx = min(idx * det_freq, T - 1)
            frame_bgr = rgb_to_bgr(frames[frame_idx].astype(np.uint8))
            if self.cfg.CROP_FACE:
                box = self._detect_face_box(frame_bgr)
            else:
                h0, w0 = frame_bgr.shape[:2]
                box = np.array([0, 0, w0, h0], dtype=np.int32)
            face_boxes.append(box)

        face_boxes = np.stack(face_boxes, axis=0)  # (num_det, 4)

        roi_frames = np.zeros((T, out_h, out_w, 3), dtype=np.float32)
        used_boxes = np.zeros((T, 4), dtype=np.int32)

        for i in range(T):
            ref = (i // det_freq) if self.cfg.DYNAMIC_DETECTION else 0
            x, y, bw, bh = face_boxes[ref]

            frame = frames[i]
            h0, w0 = frame.shape[:2]

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w0, x + bw)
            y2 = min(h0, y + bh)

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                crop = frame
                x1, y1, x2, y2 = 0, 0, w0, h0

            resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)

            roi_frames[i] = resized.astype(np.float32)
            used_boxes[i] = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.int32)

        meta = {
            "boxes_xywh": used_boxes,
            "method": "haar",
        }
        return roi_frames, meta
#----------------------------------------------------------------------------------------
# Mediapipe ROI extractor (Matthew)
#---------------------------------------------------------------------------------------

import mediapipe as mp

# --- MediaPipe landmark index constants ---
LANDMARK_REGIONS = {
    "glabella": [9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164,
                 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175],
    "forehead": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                 162, 21, 54, 103, 67, 109],
    "left_cheek": [116, 117, 118, 119, 120, 121, 128, 126, 142,
                   36, 205, 187, 123, 50, 101, 100, 47, 114],
    "right_cheek": [345, 346, 347, 348, 349, 350, 357, 355, 371,
                    266, 425, 411, 352, 280, 330, 329, 277, 343],
    "nose": [1, 2, 3, 4, 5, 6, 168, 197, 195, 94, 19, 141, 370],
    "full_face": list(range(0, 468)),  # all landmarks → bounding box
}


@dataclass
class MediaPipeROIConfig:
    W: int
    H: int
    region: str = "glabella"          # which region to crop
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    fallback_to_full_frame: bool = True  # if no face, use full frame


class MediaPipeROIExtractor(ROIExtractorBase):

    def __init__(self, cfg: MediaPipeROIConfig):
        self.cfg = cfg
        # Don't initialize face_mesh here — can't be pickled for multiprocessing
        self.landmark_indices = LANDMARK_REGIONS[cfg.region.lower()]

    def _get_face_mesh(self):
        # Create fresh instance each call — safe for multiprocessing
        return mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=self.cfg.min_detection_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
            static_image_mode=True,
            refine_landmarks=False
        )

    def _get_bounding_box(self, landmarks, h, w):
        pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)]for i in self.landmark_indices if i < len(landmarks)])
        if len(pts) == 0:
            return None
        x1 = max(0, pts[:, 0].min())
        y1 = max(0, pts[:, 1].min())
        x2 = min(w, pts[:, 0].max())
        y2 = min(h, pts[:, 1].max())

        # Add padding — glabella landmarks cluster tightly, pad to get a usable region
        pad_x = max(10, int((x2 - x1) * 0.5))
        pad_y = max(5, int((y2 - y1) * 0.1))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y1 + (y2 - y1) // 2)

        return x1, y1, x2, y2

    
    def extract(self, frames: np.ndarray) -> Tuple[np.ndarray, Dict]:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected (T, H, W, 3). Got {frames.shape}")

        T, h0, w0, _ = frames.shape
        out_h, out_w = self.cfg.H, self.cfg.W
        roi_frames = np.zeros((T, out_h, out_w, 3), dtype=np.float32)
        used_boxes = np.zeros((T, 4), dtype=np.int32)
        fallback_count = 0
        
        # Create face_mesh fresh inside extract — lives only in this worker process
        face_mesh = self._get_face_mesh()

        for i in range(T):
            frame = frames[i].astype(np.uint8)
            results = face_mesh.process(frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                box = self._get_bounding_box(landmarks, h0, w0)
            else:
             box = None


            if box is None:
                fallback_count += 1
                if self.cfg.fallback_to_full_frame:
                    crop = frame
                    box = (0, 0, w0, h0)
                else:
                    roi_frames[i] = roi_frames[i - 1] if i > 0 else 0
                    continue
            else:
                x1, y1, x2, y2 = box
                crop = frame[y1:y2, x1:x2]

            resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
            roi_frames[i] = resized.astype(np.float32)
            x1, y1, x2, y2 = box
            used_boxes[i] = [x1, y1, x2 - x1, y2 - y1]
        # Add this right before the return statement:
        

        face_mesh.close()  # explicitly release resources when done

        meta = {
            "boxes_xywh": used_boxes,
            "method": "mediapipe",
            "region": self.cfg.region,
        }
        print(f"[MediaPipe] Fallback rate: {fallback_count}/{T} frames "f"({100*fallback_count/T:.1f}%)")
        return roi_frames, meta

# ----------------------------
# Factory function
# ----------------------------
def build_roi_extractor(config_preprocess) -> ROIExtractorBase:
    method = getattr(config_preprocess, "ROI_METHOD", "haar").lower()
    
    if method == "haar":
        cfg = ROIConfig(
            W=int(config_preprocess.W),
            H=int(config_preprocess.H),
            CROP_FACE=bool(getattr(config_preprocess, "CROP_FACE", True)),
            DYNAMIC_DETECTION=bool(getattr(config_preprocess, "DYNAMIC_DETECTION", True)),
            DYNAMIC_DETECTION_FREQUENCY=int(getattr(config_preprocess, "DYNAMIC_DETECTION_FREQUENCY", 30)),
            LARGE_FACE_BOX=bool(getattr(config_preprocess, "LARGE_FACE_BOX", False)),
            LARGE_BOX_COEF=float(getattr(config_preprocess, "LARGE_BOX_COEF", 1.0)),
        )
        return HaarCascadeROIExtractor(cfg)

    elif method == "mediapipe":
        cfg = MediaPipeROIConfig(
            W=int(config_preprocess.W),
            H=int(config_preprocess.H),
            region=str(getattr(config_preprocess, "ROI_REGION", "glabella")),
            min_detection_confidence=float(getattr(config_preprocess, "MIN_DETECTION_CONFIDENCE", 0.5)),
            min_tracking_confidence=float(getattr(config_preprocess, "MIN_TRACKING_CONFIDENCE", 0.5)),
        )
        return MediaPipeROIExtractor(cfg)

    else:
        raise ValueError(f"Unknown ROI_METHOD: '{method}'. Choose from: haar, mediapipe")