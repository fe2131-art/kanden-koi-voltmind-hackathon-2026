from typing import List, Optional

from .schema import (
    AudioCue,
    DetectedObject,
    Hazard,
    Observation,
    PerceptionIR,
    UnobservedRegion,
)

# =========================
# Perception module (ハザード推定専用エンジン)
# =========================


class Perceiver:
    """
    ハザード推定・未確認領域推定の専門エンジン。
    vision_node / audio_node が収集した検出結果を受け取って PerceptionIR を組み立てる。
    """

    def __init__(self):
        """初期化（VLM・YOLO・音声処理は modality_nodes.py に移管）。"""
        pass

    def estimate(
        self,
        obs: Observation,
        objects: List[DetectedObject],
        audio_cues: List[AudioCue],
        vision_description: Optional[str] = None,
    ) -> PerceptionIR:
        """
        fuse_modalities ノードから呼ばれる唯一のエントリポイント。
        """
        unobs = self._infer_unobserved(obs, objects, audio_cues)
        hazards = self._infer_hazards(objects, audio_cues)

        return PerceptionIR(
            obs_id=obs.obs_id,
            camera_pose=obs.camera_pose,
            objects=objects,
            hazards=hazards,
            unobserved=unobs,
            audio=audio_cues,
            vision_description=vision_description,
        )

    def _infer_unobserved(
        self, obs: Observation, objects: List[DetectedObject], audio: List[AudioCue]
    ) -> List[UnobservedRegion]:
        """未確認領域を推定する。"""
        regions = [
            UnobservedRegion(
                region_id="blind_left",
                description="左側の死角（未確認領域）",
                risk=0.4,
                suggested_pan_deg=obs.camera_pose.pan_deg - 30,
            ),
            UnobservedRegion(
                region_id="blind_right",
                description="右側の死角（未確認領域）",
                risk=0.4,
                suggested_pan_deg=obs.camera_pose.pan_deg + 30,
            ),
            UnobservedRegion(
                region_id="blind_back",
                description="背後の死角（未確認領域）",
                risk=0.3,
                suggested_pan_deg=obs.camera_pose.pan_deg + 180,
            ),
        ]
        # 音声キューに基づいてリスクを上昇させる
        for c in audio:
            if c.direction == "right":
                for r in regions:
                    if r.region_id == "blind_right":
                        r.risk = min(1.0, r.risk + 0.3)
            if c.direction == "left":
                for r in regions:
                    if r.region_id == "blind_left":
                        r.risk = min(1.0, r.risk + 0.3)
        return regions

    def _infer_hazards(
        self, objects: List[DetectedObject], audio: List[AudioCue]
    ) -> List[Hazard]:
        """検出結果からハザードを推定する。"""
        hazards: List[Hazard] = []
        labels = {o.label for o in objects}

        # 検出オブジェクトに基づくハザード
        if "person" in labels:
            hazards.append(
                Hazard(
                    hazard_type="human_present",
                    confidence=0.6,
                    related_objects=["person"],
                    evidence="person detected",
                )
            )

        if "foreground_object" in labels:
            hazards.append(
                Hazard(
                    hazard_type="unidentified_foreground_object",
                    confidence=0.5,
                    related_objects=["foreground_object"],
                    evidence="foreground object detected in image",
                )
            )

        # 音声ベースのハザード
        if any(c.cue == "vehicle_approaching" for c in audio):
            hazards.append(
                Hazard(
                    hazard_type="vehicle_possible_blindspot",
                    confidence=0.6,
                    related_objects=[],
                    evidence="audio cue suggests vehicle",
                )
            )

        return hazards

    def run(self, obs: Observation) -> PerceptionIR:
        """
        後方互換ラッパー（test_e2e.py が直接呼ぶ場合に使用）。
        """
        from .modality_nodes import AudioAnalyzer

        objects = []
        audio_cues = AudioAnalyzer().analyze(obs.audio_text)
        return self.estimate(obs, objects, audio_cues)
