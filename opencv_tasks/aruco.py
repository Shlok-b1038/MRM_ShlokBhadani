#!/usr/bin/env python3
"""
Real-time ArUco marker detection + pose / depth estimation with OpenCV.

Detects 4x4, 5x5 and 6x6 ArUco markers simultaneously (e.g. generated at
https://chev.me/arucogen) and overlays, for every marker:
    - outline, ID and dictionary
    - 3D axes (X red, Y green, Z blue)
    - position  (X, Y, Z) in cm, relative to the camera
    - depth (Z) and straight-line distance
    - orientation as roll / pitch / yaw in degrees

Usage:
    python aruco_pose.py --marker-size 50
    python aruco_pose.py --marker-size 100 --camera 1 --calib calib.npz

Keys:  q / ESC = quit,   s = save screenshot

Requirements:
    pip install opencv-contrib-python numpy
"""

import argparse
import time

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Dictionaries
# --------------------------------------------------------------------------- #
# Markers from different grid sizes (4x4 / 5x5 / 6x6) can never be confused
# with each other, so we simply run one detector per grid size.
# Within a grid size, the 50/100/250/1000 dictionaries are nested (the first
# 50 markers of the 250 dictionary *are* the 50 dictionary), so a single
# dictionary per size covers all IDs up to its number.
DICT_TABLE = {
    "4x4": {50: cv2.aruco.DICT_4X4_50, 100: cv2.aruco.DICT_4X4_100,
            250: cv2.aruco.DICT_4X4_250, 1000: cv2.aruco.DICT_4X4_1000},
    "5x5": {50: cv2.aruco.DICT_5X5_50, 100: cv2.aruco.DICT_5X5_100,
            250: cv2.aruco.DICT_5X5_250, 1000: cv2.aruco.DICT_5X5_1000},
    "6x6": {50: cv2.aruco.DICT_6X6_50, 100: cv2.aruco.DICT_6X6_100,
            250: cv2.aruco.DICT_6X6_250, 1000: cv2.aruco.DICT_6X6_1000},
}

AXIS_COLORS = {"4x4": (0, 255, 0), "5x5": (255, 200, 0), "6x6": (0, 165, 255)}


# --------------------------------------------------------------------------- #
# OpenCV version compatibility (API changed in 4.7)
# --------------------------------------------------------------------------- #
class Detector:
    """Thin wrapper hiding the OpenCV <4.7 / >=4.7 ArUco API difference."""

    def __init__(self, dict_id):
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        if hasattr(cv2.aruco, "ArucoDetector"):            # OpenCV >= 4.7
            self.params = cv2.aruco.DetectorParameters()
            self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self._det = cv2.aruco.ArucoDetector(self.dictionary, self.params)
        else:                                              # OpenCV < 4.7
            self.params = cv2.aruco.DetectorParameters_create()
            self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self._det = None

    def detect(self, gray):
        if self._det is not None:
            corners, ids, _ = self._det.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=self.params)
        return corners, ids


# --------------------------------------------------------------------------- #
# Camera intrinsics
# --------------------------------------------------------------------------- #
def load_calibration(path):
    """Load camera_matrix + dist_coeffs from .npz or OpenCV .yml/.xml/.json."""
    if path.lower().endswith(".npz"):
        data = np.load(path)
        return data["camera_matrix"], data["dist_coeffs"]
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Cannot open calibration file: {path}")
    cam = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeffs").mat()
    fs.release()
    if cam is None or dist is None:
        raise ValueError("Calibration file must contain 'camera_matrix' and 'dist_coeffs'")
    return cam, dist


def approximate_intrinsics(width, height, hfov_deg=65.0):
    """
    Rough pinhole model for an uncalibrated webcam (typical laptop HFOV ~60-70°).
    Good enough for a demo; for accurate distances calibrate your camera.
    """
    fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    cam = np.array([[fx, 0, width / 2.0],
                    [0, fx, height / 2.0],
                    [0, 0, 1]], dtype=np.float64)
    return cam, np.zeros((5, 1), dtype=np.float64)


# --------------------------------------------------------------------------- #
# Pose helpers
# --------------------------------------------------------------------------- #
def marker_object_points(size_m):
    """3D corners of a square marker centred at the origin (TL, TR, BR, BL)."""
    h = size_m / 2.0
    return np.array([[-h,  h, 0],
                     [ h,  h, 0],
                     [ h, -h, 0],
                     [-h, -h, 0]], dtype=np.float32)


def estimate_pose(corners, obj_pts, cam, dist):
    """solvePnP-based pose (estimatePoseSingleMarkers is deprecated/removed)."""
    img_pts = corners.reshape(4, 2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, cam, dist,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, cam, dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
    return (rvec, tvec) if ok else (None, None)


def rotation_to_euler_deg(rvec):
    """Rodrigues vector -> (roll, pitch, yaw) in degrees (X, Y, Z rotations)."""
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def put_text(img, text, org, color=(255, 255, 255), scale=0.5):
    """Text with a dark outline so it stays readable on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="ArUco detection + pose estimation")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--marker-size", type=float, default=50.0,
                    help="printed side length of the black marker square, in mm "
                         "(default 50). MUST match your printed tag.")
    ap.add_argument("--dict-size", type=int, choices=[50, 100, 250, 1000], default=250,
                    help="dictionary size per grid type (default 250; "
                         "arucogen's default is 50, which is covered)")
    ap.add_argument("--calib", type=str, default=None,
                    help="calibration file (.npz or .yml) with camera_matrix & dist_coeffs")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--hfov", type=float, default=65.0,
                    help="assumed horizontal FOV in degrees if no calibration given")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.calib:
        cam, dist = load_calibration(args.calib)
        calib_note = "calibrated"
    else:
        cam, dist = approximate_intrinsics(w, h, args.hfov)
        calib_note = "UNCALIBRATED (approx. intrinsics)"
        print("[warn] No calibration file given - using approximate intrinsics; "
              "distances will be only roughly correct.")
    print(f"[info] Resolution {w}x{h}, marker size {args.marker_size} mm, {calib_note}")

    size_m = args.marker_size / 1000.0
    obj_pts = marker_object_points(size_m)
    detectors = {name: Detector(ids[args.dict_size]) for name, ids in DICT_TABLE.items()}

    prev = time.time()
    fps = 0.0
    shot = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[error] Failed to read frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        n_found = 0

        for name, det in detectors.items():
            corners, ids = det.detect(gray)
            if ids is None:
                continue

            color = AXIS_COLORS[name]
            for c, mid in zip(corners, ids.flatten()):
                rvec, tvec = estimate_pose(c, obj_pts, cam, dist)
                if rvec is None:
                    continue
                n_found += 1

                # outline
                pts = c.reshape(4, 2).astype(int)
                cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
                cv2.circle(frame, tuple(pts[0]), 5, (0, 0, 255), -1)  # corner 0

                # 3D axes (length = half the marker size)
                cv2.drawFrameAxes(frame, cam, dist, rvec, tvec, size_m * 0.6, 2)

                # numbers
                x, y, z = tvec.flatten() * 100.0          # cm
                dist_cm = float(np.linalg.norm(tvec)) * 100.0
                roll, pitch, yaw = rotation_to_euler_deg(rvec)

                tx = int(pts[:, 0].min())
                ty = int(pts[:, 1].min()) - 10
                lines = [
                    f"ID {mid}  [{name}]",
                    f"Pos X:{x:6.1f} Y:{y:6.1f} Z:{z:6.1f} cm",
                    f"Depth(Z): {z:.1f} cm  Dist: {dist_cm:.1f} cm",
                    f"R:{roll:6.1f} P:{pitch:6.1f} Y:{yaw:6.1f} deg",
                ]
                # draw above the marker if there is room, otherwise below it
                if ty - 18 * len(lines) < 0:
                    ty = int(pts[:, 1].max()) + 20
                    step = 18
                else:
                    ty = ty - 18 * (len(lines) - 1)
                    step = 18
                for i, line in enumerate(lines):
                    put_text(frame, line, (max(tx, 5), ty + i * step),
                             color if i == 0 else (255, 255, 255))

        # HUD
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now
        put_text(frame, f"FPS: {fps:4.1f}   markers: {n_found}   {calib_note}",
                 (10, 22), (0, 255, 255), 0.55)
        put_text(frame, "q/ESC: quit   s: screenshot", (10, h - 12), (200, 200, 200), 0.45)

        cv2.imshow("ArUco pose estimation", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            fn = f"aruco_shot_{shot:03d}.png"
            cv2.imwrite(fn, frame)
            print(f"[info] saved {fn}")
            shot += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
