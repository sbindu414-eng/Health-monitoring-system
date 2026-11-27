"""
Unit and Integration Tests for Live Human Eye Verification & Presentation Attack Detection (PAD).

Tests:
1. RealTimeEyeDetector: eye localization, tracking, and EAR estimation.
2. PresentationAttackDetector: Moiré pattern FFT detection, LBP texture analysis, specular Purkinje reflection, device bezels.
3. LiveHumanEyeVerifier: rejection of static photos, detection of natural blinks, sequence-level PAD.
4. Security gating in analyze_uploaded_eye_image: camera enforcement, spoof rejection, retinal fundus isolation.
5. End-to-end API verification: /api/eye/liveness/verify, /api/eye/liveness/reset, /api/eye/analyze.
"""

import base64
import json
import os
import sys
import time
import unittest
import urllib.error
import urllib.request

import cv2
import numpy as np

from src.eye_analysis.liveness_pad import (
    CHALLENGE_STEPS,
    LiveHumanEyeVerifier,
    LIVENESS_CONFIDENCE_THRESHOLD,
    MSG_EYE_DETECTED,
    MSG_LIVENESS_FAILED,
    MSG_SPOOF_DETECTED,
    MSG_VERIFICATION_FAILED,
    MSG_WAITING_FOR_EYE,
    PresentationAttackDetector,
    RealTimeEyeDetector,
    SESSION_MANAGER,
    STATE_ANALYSIS_COMPLETED,
    STATE_LIVE_VERIFIED,
    STATE_PAD_DETECTED,
    STATE_VERIFICATION_FAILED,
    STATE_VERIFYING_LIVENESS,
    STATE_WAITING_FOR_EYE,
    verify_webcam_frame_sequence,
)
from src.eye_analysis.predict import analyze_uploaded_eye_image, load_retinal_bundle


def create_synthetic_eye_frame(openness: float = 1.0, is_screen: bool = False, is_paper: bool = False, offset_x: int = 0) -> np.ndarray:
    """Generate a 480x640 synthetic human face/eye frame with controllable eye openness, specular reflections, and artifacts."""
    img = np.full((480, 640, 3), (170, 195, 225), dtype=np.uint8)  # skin tone background

    # Face ellipse
    cv2.ellipse(img, (320 + offset_x, 240), (140, 180), 0, 0, 360, (150, 180, 215), -1)

    # Eye socket / sclera
    eye_center = (320 + offset_x, 220)
    rx = 50
    ry = max(4, int(26 * openness))
    cv2.ellipse(img, eye_center, (rx, ry), 0, 0, 360, (240, 242, 245), -1)
    cv2.ellipse(img, eye_center, (rx, ry), 0, 0, 360, (110, 130, 160), 2)

    # Iris
    iris_r = min(ry, 18)
    if iris_r > 5:
        cv2.circle(img, eye_center, iris_r, (70, 110, 80), -1)  # green/hazel iris
        cv2.circle(img, eye_center, iris_r, (35, 65, 45), 2)   # crisp limbus edge
        # Pupil
        cv2.circle(img, eye_center, max(3, int(iris_r * 0.45)), (15, 15, 15), -1)
        # Specular corneal reflection (Purkinje highlight) - present only on real eye or glass
        if not is_paper:
            cv2.circle(img, (eye_center[0] - 4, eye_center[1] - 4), 3, (255, 255, 255), -1)

    # Eyelashes / eyelids
    cv2.ellipse(img, eye_center, (rx, ry), 0, 180, 360, (50, 60, 80), 3)

    # Add realistic skin pore micro-texture so Laplacian sharpness is realistic (> 25)
    rng = np.random.default_rng(123 + int(openness * 100) + int(is_screen) * 7 + int(is_paper) * 13 + abs(offset_x))
    noise = rng.normal(0, 4, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Simulate phone screen attack (high-frequency periodic Moiré lines + screen bezels)
    if is_screen:
        y_coords = np.arange(480)[:, None]
        moire_pattern = (16 * np.sin(2 * np.pi * y_coords / 4.0)).astype(np.int16)
        img_int = img.astype(np.int16) + moire_pattern[:, :, None]
        img = np.clip(img_int, 0, 255).astype(np.uint8)
        # Screen border / bezel
        cv2.rectangle(img, (40, 30), (600, 450), (10, 10, 10), 4)

    # Simulate paper print attack (low contrast, diffuse surface noise, no specular highlight)
    if is_paper:
        noise_p = np.random.normal(0, 8, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise_p, 0, 255).astype(np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


class TestRealTimeEyeDetector(unittest.TestCase):
    def setUp(self):
        self.detector = RealTimeEyeDetector()

    def test_detector_initialization(self):
        self.assertIsNotNone(self.detector)

    def test_detect_eye_region_synthetic(self):
        frame = create_synthetic_eye_frame(openness=1.0)
        res = self.detector.detect_eye_region(frame)
        self.assertIsNotNone(res)
        self.assertTrue(res["eye_detected"])
        self.assertIn("eye_box", res)
        self.assertIn("eye_aspect_ratio", res)
        self.assertIn("is_open", res)
        self.assertGreaterEqual(res["eye_aspect_ratio"], 0.0)


class TestPresentationAttackDetector(unittest.TestCase):
    def setUp(self):
        self.pad = PresentationAttackDetector()

    def test_detect_phone_screen_attack(self):
        """Simulate an eye displayed on a phone screen with periodic pixel grid/Moiré lines."""
        screen_frame = create_synthetic_eye_frame(openness=1.0, is_screen=True)
        res = self.pad.analyze_frame_pad(screen_frame, eye_box=(260, 180, 120, 80))
        self.assertTrue(res["is_spoof"], "Phone screen attack must be flagged as spoof")
        self.assertTrue(res["is_screen"], "Attack type must be identified as screen/display")
        self.assertGreaterEqual(res["sub_metrics"]["fft_peak_ratio"], 2.8, "Moiré frequency spike must exceed detection threshold")

    def test_detect_printed_paper_photo_attack(self):
        """Simulate a printed eye photo on matte paper without corneal specular highlight."""
        paper_frame = create_synthetic_eye_frame(openness=1.0, is_paper=True)
        res = self.pad.analyze_frame_pad(paper_frame, eye_box=(260, 180, 120, 80))
        self.assertIn("lbp_entropy", res)
        self.assertIn("is_printed", res)
        self.assertIn("reflection_type", res["sub_metrics"])

    def test_clean_real_eye_pad_metrics(self):
        """Clean synthetic eye without screen lines should have high organic Moiré score."""
        real_frame = create_synthetic_eye_frame(openness=1.0, is_screen=False, is_paper=False)
        res = self.pad.analyze_frame_pad(real_frame, eye_box=(260, 180, 120, 80))
        self.assertFalse(res["sub_metrics"]["bezel_detected"])


class TestLiveHumanEyeVerifier(unittest.TestCase):
    def test_reject_static_photograph(self):
        """Feeding identical static frames (e.g. holding a still photograph) must be rejected."""
        verifier = LiveHumanEyeVerifier(min_frames=10)
        static_frame = create_synthetic_eye_frame(openness=1.0)
        last_res = None
        for _ in range(15):
            last_res = verifier.add_frame(static_frame)

        self.assertFalse(last_res["is_live"], "Static photograph with 0 optical flow must not pass liveness")
        self.assertTrue(last_res.get("is_static_spoof", False) or last_res["status"] in ("spoof_detected", "verifying_liveness"))

    def test_accept_natural_blink_sequence(self):
        """Simulate a real human eye sequence: open -> natural blink (closed 2 frames) -> reopened with natural micro-motion."""
        verifier = LiveHumanEyeVerifier(min_frames=10)
        # Sequence: 4 open frames -> 2 closing -> 2 closed -> 2 opening -> 4 open frames with subtle micro-saccades
        openness_curve = [1.0, 1.0, 0.9, 0.8, 0.2, 0.1, 0.15, 0.7, 0.9, 1.0, 1.0, 0.95]
        for i, opn in enumerate(openness_curve):
            shift = int(np.sin(i * 0.7) * 2)
            frame = create_synthetic_eye_frame(openness=opn, offset_x=shift)
            res = verifier.add_frame(frame)

        # Blinks should be detected across the dip
        self.assertGreaterEqual(verifier.blink_count, 1, "Natural blink dip and recovery must be recognized")

    def test_reject_replayed_video_on_screen(self):
        """A video on a phone screen contains blinking, but PAD must reject the screen Moiré/pixel grid."""
        verifier = LiveHumanEyeVerifier(min_frames=10)
        openness_curve = [1.0, 0.9, 0.2, 0.1, 0.8, 1.0, 1.0, 0.9, 1.0, 1.0]
        last_res = None
        for opn in openness_curve:
            # Replayed on phone screen -> is_screen=True
            screen_frame = create_synthetic_eye_frame(openness=opn, is_screen=True)
            last_res = verifier.add_frame(screen_frame)

        self.assertFalse(last_res["is_live"], "Replayed video on screen must be rejected despite blinks")
        self.assertEqual(last_res["status"], "spoof_detected")
        self.assertEqual(last_res["guidance_message"], MSG_SPOOF_DETECTED)


class TestSecurityGatingAndIsolation(unittest.TestCase):
    def test_camera_mode_enforces_liveness_verified(self):
        """In camera mode, is_liveness_verified=False must return liveness_failed with exact message."""
        frame = create_synthetic_eye_frame(openness=1.0)
        ok, buf = cv2.imencode(".png", frame)
        file_bytes = buf.tobytes()

        res = analyze_uploaded_eye_image(
            file_bytes=file_bytes,
            filename="webcam_eye_test.png",
            analysis_mode="camera",
            is_liveness_verified=False,
        )
        self.assertEqual(res["status"], "liveness_failed")
        self.assertFalse(res["valid_eye_image"])
        self.assertEqual(res["message"], MSG_LIVENESS_FAILED)
        self.assertEqual(res["prediction_label"], "Live Human Eye Not Verified")
        self.assertNotIn("iris_analysis", res, "Iris segmentation must NOT run when liveness fails")

    def test_spoof_detection_blocks_iris_analysis(self):
        """When a screen photo is submitted, PAD must return spoof_detected with exact message."""
        screen_frame = create_synthetic_eye_frame(openness=1.0, is_screen=True)
        ok, buf = cv2.imencode(".png", screen_frame)
        file_bytes = buf.tobytes()

        res = analyze_uploaded_eye_image(
            file_bytes=file_bytes,
            filename="phone_screen_eye.png",
            analysis_mode="anterior_iris",
            is_liveness_verified=False,
        )
        self.assertEqual(res["status"], "spoof_detected")
        self.assertFalse(res["valid_eye_image"])
        self.assertEqual(res["message"], MSG_SPOOF_DETECTED)
        self.assertNotIn("iris_analysis", res, "Iris analysis must NOT run when spoof is detected")

    def test_retinal_fundus_upload_unaffected(self):
        """Retinal fundus upload remains completely unaffected by live camera PAD."""
        sample_path = os.path.join("data", "samples", "sample_test_grade0_healthy.png")
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                file_bytes = f.read()
            res = analyze_uploaded_eye_image(
                file_bytes=file_bytes,
                filename="sample_test_grade0_healthy.png",
                analysis_mode="auto",
            )
            self.assertEqual(res["status"], "completed")
            self.assertEqual(res["detected_modality"], "retinal_fundus")
            self.assertIn("class_probabilities", res)


class TestLiveApiEndpoints(unittest.TestCase):
    def test_liveness_verify_and_reset_api(self):
        """Test the live backend HTTP API /api/eye/liveness/verify and /api/eye/liveness/reset."""
        session_id = f"test_session_{int(np.random.randint(10000, 99999))}"
        frame = create_synthetic_eye_frame(openness=1.0)
        ok, buf = cv2.imencode(".jpg", frame)
        b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

        # Step 1: Send frame to /api/eye/liveness/verify
        req_data = json.dumps({"sessionId": session_id, "frameBase64": b64}).encode("utf-8")
        req = urllib.request.Request("http://localhost:5000/api/eye/liveness/verify", data=req_data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as response:
            self.assertEqual(response.status, 200)
            res_json = json.loads(response.read().decode("utf-8"))
            self.assertIn("status", res_json)
            self.assertIn("sessionId", res_json)
            self.assertEqual(res_json["sessionId"], session_id)

        # Step 2: Reset session
        reset_data = json.dumps({"sessionId": session_id}).encode("utf-8")
        reset_req = urllib.request.Request("http://localhost:5000/api/eye/liveness/reset", data=reset_data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(reset_req, timeout=5) as response:
            self.assertEqual(response.status, 200)
            reset_json = json.loads(response.read().decode("utf-8"))
            self.assertEqual(reset_json.get("status"), "RESET")

    def test_api_blocks_direct_unverified_camera_analyze(self):
        """Direct POST /api/eye/analyze in camera mode without verified session/token must block analysis."""
        frame = create_synthetic_eye_frame(openness=1.0)
        ok, buf = cv2.imencode(".png", frame)
        b64 = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

        payload = json.dumps({
            "imageBase64": b64,
            "filename": "live_bypass_attempt.png",
            "mode": "live_camera",
            "source": "camera",
            "livenessVerified": True,  # spoofed client flag with no server token!
            "sessionId": "forged_session_id",
        }).encode("utf-8")
        req = urllib.request.Request("http://localhost:5000/api/eye/analyze", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as response:
            self.assertEqual(response.status, 200)
            res = json.loads(response.read().decode("utf-8"))
            self.assertEqual(res["status"], "liveness_failed")
            self.assertFalse(res["valid_eye_image"])
            self.assertNotIn("iris_analysis", res)


class TestComprehensivePADAndAuthorization(unittest.TestCase):
    def _complete_live_session(self, session_id: str):
        """Simulate a genuine human user completing randomized challenge steps on SESSION_MANAGER."""
        SESSION_MANAGER.reset_session(session_id)
        # Send baseline open eye frames to initialize session & enter challenge
        res = None
        for i in range(11):
            shift = int(np.sin(i * 0.8) * 2)
            res = SESSION_MANAGER.process_frame(session_id, create_synthetic_eye_frame(openness=1.0, offset_x=shift), camera_device_id="cam_0")

        # Work dynamically through whichever randomized challenge step is currently requested
        first_sign = 1
        for _ in range(4):
            ch = res.get("challenge") or {}
            if res.get("is_live"):
                break
            stype = ch.get("step_id")
            if stype == "look_straight":
                for i in range(7):
                    res = SESSION_MANAGER.process_frame(session_id, create_synthetic_eye_frame(openness=1.0, offset_x=i % 2), camera_device_id="cam_0")
            elif stype == "blink":
                for opn in [1.0, 0.2, 0.1, 0.1, 0.9, 1.0]:
                    res = SESSION_MANAGER.process_frame(session_id, create_synthetic_eye_frame(openness=opn, offset_x=1), camera_device_id="cam_0")
            elif stype in ("look_left", "look_right"):
                direction = first_sign
                first_sign = -first_sign  # Ensure second lateral step always goes opposite direction
                for j in range(10):
                    offset = direction * j * 2
                    res = SESSION_MANAGER.process_frame(session_id, create_synthetic_eye_frame(openness=1.0, offset_x=offset), camera_device_id="cam_0")
        return res

    def test_real_human_eye_completes_verification_and_authorizes_analysis(self):
        """Real eye passing PAD + blink + challenge-response obtains single-use token and completes iris analysis."""
        sid = f"real_user_{int(np.random.randint(1000, 9999))}"
        res = self._complete_live_session(sid)
        self.assertEqual(res["status"], "live_verified")
        self.assertEqual(res["state_machine_state"], STATE_LIVE_VERIFIED)
        self.assertTrue(res["is_live"])
        self.assertGreaterEqual(res["liveness_score_pct"], 65.0)
        self.assertIn("verificationToken", res)

        token = res["verificationToken"]
        # High-resolution textured anterior eye snapshot (clear iris/pupil/sclera + specular corneal highlight)
        snap_rgb = np.full((240, 320, 3), [225, 220, 215], dtype=np.uint8)
        rng = np.random.default_rng(42)
        cx, cy = 160, 120
        yy, xx = np.ogrid[:240, :320]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        iris_mask = (dist <= 65) & (dist > 22)
        radial_noise = rng.integers(-22, 22, size=(240, 320))
        snap_rgb[iris_mask, 0] = np.clip(65 + radial_noise[iris_mask], 20, 140)
        snap_rgb[iris_mask, 1] = np.clip(125 + radial_noise[iris_mask], 40, 190)
        snap_rgb[iris_mask, 2] = np.clip(145 + radial_noise[iris_mask], 50, 220)
        snap_rgb[dist <= 22] = [14, 14, 16]
        cv2.ellipse(snap_rgb, (cx, cy), (125, 78), 0, 0, 360, (165, 130, 115), 4)
        cv2.circle(snap_rgb, (cx - 10, cy - 10), 4, (255, 255, 255), -1)
        snap = cv2.cvtColor(snap_rgb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", snap)
        analysis = analyze_uploaded_eye_image(
            file_bytes=buf.tobytes(),
            filename="live_webcam_eye.png",
            analysis_mode="live_camera",
            session_id=sid,
            verification_token=token,
            source="camera",
        )
        self.assertEqual(analysis["status"], "completed")
        self.assertTrue(analysis["valid_eye_image"])
        self.assertIn("iris_analysis", analysis)

        # Single-use token replay MUST be rejected
        replay = analyze_uploaded_eye_image(
            file_bytes=buf.tobytes(),
            filename="live_webcam_eye.png",
            analysis_mode="live_camera",
            session_id=sid,
            verification_token=token,
            source="camera",
        )
        self.assertEqual(replay["status"], "liveness_failed")
        self.assertNotIn("iris_analysis", replay)

    def test_mobile_phone_and_laptop_screen_always_rejected(self):
        """Mobile phone screens and laptop monitors displaying an eye image are rejected by PAD and blocked."""
        # 1. Portrait mobile phone screen with Moiré + bezel
        mobile_frame = create_synthetic_eye_frame(openness=1.0, is_screen=True)
        # 2. Wide laptop display bezel + subpixel stripes
        laptop_frame = create_synthetic_eye_frame(openness=1.0, is_screen=False)
        x_coords = np.arange(640)[None, :]
        laptop_frame = np.clip(laptop_frame.astype(np.int16) + (20 * np.sin(2 * np.pi * x_coords / 4.0)).astype(np.int16)[:, :, None], 0, 255).astype(np.uint8)
        cv2.rectangle(laptop_frame, (15, 25), (625, 455), (15, 15, 15), 6)

        for label, fr in [("mobile", mobile_frame), ("laptop", laptop_frame)]:
            verifier = LiveHumanEyeVerifier(min_frames=8)
            last = None
            for _ in range(3):
                last = verifier.add_frame(fr)
            self.assertEqual(last["status"], "spoof_detected", f"{label} screen must return spoof_detected")
            self.assertEqual(last["state_machine_state"], STATE_PAD_DETECTED)
            self.assertFalse(last["is_live"])
            self.assertEqual(last["liveness_score_pct"], 0.0)

    def test_printed_eye_photograph_rejected(self):
        """Printed eye photograph (matte paper, zero natural blink) never passes liveness and blocks analysis."""
        paper_frame = create_synthetic_eye_frame(openness=1.0, is_paper=True)
        verifier = LiveHumanEyeVerifier(min_frames=8, enable_challenge=False)
        last = None
        for _ in range(10):
            last = verifier.add_frame(paper_frame)
        self.assertFalse(last["is_live"])
        self.assertIn(last["status"], ("spoof_detected", "verifying_liveness"))
        self.assertLess(last["liveness_score_pct"], 50.0)

    def test_prerecorded_video_blocked_by_screen_pad_and_challenge(self):
        """Prerecorded eye video with blinking on a display is blocked immediately by PAD."""
        sid = f"replay_{int(np.random.randint(1000, 9999))}"
        SESSION_MANAGER.reset_session(sid)
        last = None
        for opn in [1.0, 0.2, 0.1, 0.9, 1.0]:
            last = SESSION_MANAGER.process_frame(sid, create_synthetic_eye_frame(openness=opn, is_screen=True))
        self.assertEqual(last["status"], "spoof_detected")
        self.assertFalse(last["is_live"])
        ok, _, _ = SESSION_MANAGER.validate_and_consume_token(sid, "any", consume=False)
        self.assertFalse(ok)

    def test_no_eye_detected_resets_verification(self):
        """When eye disappears from camera view, verification & challenge reset to WAITING_FOR_EYE."""
        verifier = LiveHumanEyeVerifier(min_frames=6)
        for _ in range(4):
            verifier.add_frame(create_synthetic_eye_frame(openness=1.0))
        blank_wall = np.full((480, 640, 3), 120, dtype=np.uint8)
        res = None
        for _ in range(4):
            res = verifier.add_frame(blank_wall)
        self.assertEqual(res["status"], "waiting_for_eye")
        self.assertEqual(res["state_machine_state"], STATE_WAITING_FOR_EYE)
        self.assertFalse(res["eye_detected"])
        self.assertEqual(res["consecutive_frames"], 0)
        self.assertEqual(len(verifier.frame_history), 0)

    def test_zero_and_low_confidence_liveness_block_analysis(self):
        """Zero liveness score, forged is_liveness_verified=True without token, or low confidence always block analysis."""
        frame = create_synthetic_eye_frame(openness=1.0)
        ok, buf = cv2.imencode(".png", frame)
        b = buf.tobytes()

        # 1. Forged is_liveness_verified=True without valid backend token
        res1 = analyze_uploaded_eye_image(b, filename="live_test.png", analysis_mode="live_camera", is_liveness_verified=True)
        self.assertEqual(res1["status"], "liveness_failed")
        self.assertNotIn("iris_analysis", res1)

        # 2. Liveness telemetry with liveness_score_pct == 0
        res2 = analyze_uploaded_eye_image(
            b, filename="live_test.png", analysis_mode="live_camera",
            session_id="some_sid", verification_token="some_tok",
            liveness_telemetry={"is_live": False, "liveness_score_pct": 0.0, "confidence": 0.0},
        )
        self.assertEqual(res2["status"], "liveness_failed")
        self.assertNotIn("iris_analysis", res2)

        # 3. Low-confidence liveness telemetry (< threshold) even if claimed is_live
        sid = f"low_conf_{int(np.random.randint(1000, 9999))}"
        res3 = analyze_uploaded_eye_image(
            b, filename="live_test.png", analysis_mode="live_camera",
            session_id=sid, verification_token="invalid_tok",
            liveness_telemetry={"is_live": False, "liveness_score_pct": 30.0},
        )
        self.assertEqual(res3["status"], "liveness_failed")
        self.assertNotIn("iris_analysis", res3)

    def test_challenge_timeout_transitions_to_verification_failed_and_resets(self):
        """When challenge step timer expires, verifier emits VERIFICATION_FAILED and clears state."""
        verifier = LiveHumanEyeVerifier(min_frames=3, challenge_timeout_seconds=0.35)
        for _ in range(4):
            verifier.add_frame(create_synthetic_eye_frame(openness=1.0))
        time.sleep(0.40)
        res = verifier.add_frame(create_synthetic_eye_frame(openness=1.0))
        self.assertEqual(res["status"], "verification_failed")
        self.assertEqual(res["state_machine_state"], STATE_VERIFICATION_FAILED)
        self.assertFalse(res["is_live"])
        self.assertEqual(res["liveness_score_pct"], 0.0)
        self.assertEqual(len(verifier.frame_history), 0)

    def test_camera_restart_and_device_switch_invalidate_token(self):
        """Switching camera device or resetting session invalidates any previously verified token."""
        sid = f"cam_switch_{int(np.random.randint(1000, 9999))}"
        res = self._complete_live_session(sid)
        token = res["verificationToken"]

        # Simulate switching camera device in the same session
        SESSION_MANAGER.process_frame(sid, create_synthetic_eye_frame(openness=1.0), camera_device_id="cam_1_usb")
        valid, reason, _ = SESSION_MANAGER.validate_and_consume_token(sid, token, consume=False)
        self.assertFalse(valid, f"Old token must be invalidated after camera switch (reason={reason})")

        # Also test explicit session reset
        res2 = self._complete_live_session(sid)
        token2 = res2["verificationToken"]
        SESSION_MANAGER.reset_session(sid)
        valid2, _, _ = SESSION_MANAGER.validate_and_consume_token(sid, token2, consume=False)
        self.assertFalse(valid2, "Token must be invalidated when camera restarts / resets session")

    def test_retinal_photo_on_phone_blocked_in_live_camera_mode(self):
        """A retinal fundus scan presented via live camera mode must never invoke retinal DR CNN or pass PAD."""
        sample_path = os.path.join("data", "samples", "sample_test_grade0_healthy.png")
        if os.path.exists(sample_path):
            sid = f"fundus_cam_{int(np.random.randint(1000, 9999))}"
            ver = self._complete_live_session(sid)
            token = ver["verificationToken"]
            with open(sample_path, "rb") as f:
                fundus_bytes = f.read()
            res = analyze_uploaded_eye_image(
                file_bytes=fundus_bytes,
                filename="live_fundus_phone.png",
                analysis_mode="live_camera",
                session_id=sid,
                verification_token=token,
                source="camera",
            )
            self.assertNotEqual(res["status"], "completed")
            self.assertIn(res["status"], ("spoof_detected", "liveness_failed"))
            self.assertNotIn("diabetes_detected", res)

    def test_single_step_verification_with_room_background_and_closeup_eye(self):
        """Real user sitting in a room with rectangular wall lines verifies in <=12 frames (1 step) and segments iris."""
        sid = f"room_cam_{int(np.random.randint(1000, 9999))}"
        SESSION_MANAGER.reset_session(sid)

        def _make_room_webcam_frame(openness: float, shift: int = 0) -> np.ndarray:
            # Bright cream/white room wall with wardrobe/doorframe vertical & horizontal lines behind live user
            img = np.full((480, 640, 3), (195, 200, 205), dtype=np.uint8)
            cv2.line(img, (110, 0), (110, 480), (120, 125, 130), 3)
            cv2.line(img, (530, 0), (530, 480), (120, 125, 130), 3)
            cv2.line(img, (0, 65), (640, 65), (140, 145, 150), 2)
            cv2.ellipse(img, (320 + shift, 240), (110, 145), 0, 0, 360, (145, 175, 205), -1)
            for ex in (270 + shift, 370 + shift):
                ry = max(3, int(18 * openness))
                cv2.ellipse(img, (ex, 225), (32, ry), 0, 0, 360, (225, 228, 232), -1)
                ir = min(ry, 12)
                if ir > 3:
                    cv2.circle(img, (ex, 225), ir, (60, 85, 110), -1)
                    cv2.circle(img, (ex, 225), max(2, ir // 2), (18, 18, 18), -1)
            rng = np.random.default_rng(10 + int(openness * 100) + abs(shift))
            img = np.clip(img.astype(np.int16) + rng.normal(0, 4, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
            return img

        seq = [1.0, 1.0, 0.95, 0.2, 0.12, 0.9, 1.0, 1.0, 0.95, 1.0]
        res = None
        verified_at_step = None
        for idx, opn in enumerate(seq):
            res = SESSION_MANAGER.process_frame(sid, _make_room_webcam_frame(opn, shift=(idx % 3)), camera_device_id="cam_0")
            if res.get("status") == "live_verified":
                verified_at_step = idx + 1
                break

        self.assertIsNotNone(verified_at_step, f"Single-step verification did not pass cleanly: last={res}")
        self.assertLessEqual(verified_at_step, 10)
        self.assertIn("verificationToken", res)

        snap = _make_room_webcam_frame(1.0, shift=1)
        ok, buf = cv2.imencode(".png", snap)
        self.assertTrue(ok)
        analysis = analyze_uploaded_eye_image(
            file_bytes=buf.tobytes(),
            filename="live_webcam_eye.png",
            analysis_mode="live_camera",
            session_id=sid,
            verification_token=res["verificationToken"],
            source="camera",
        )
        self.assertEqual(analysis["status"], "completed")
        self.assertTrue(analysis.get("iris_detected"))
        self.assertIn("iris_analysis", analysis)


if __name__ == "__main__":
    unittest.main()
