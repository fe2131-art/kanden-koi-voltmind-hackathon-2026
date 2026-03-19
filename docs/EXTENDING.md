# Safety View Agent - 拡張ガイド

Safety View Agent は拡張可能な設計になっています。新しいセンサーやロジックを追加できます。

## システムの拡張ポイント

### 1. 新しいモダリティ（センサー）の追加

最も一般的な拡張です。例として LiDAR を追加します。

#### Step 1: モダリティ処理クラスを作成

`src/safety_agent/modality_nodes.py` に追加：

```python
class LidarAnalyzer:
    """LiDAR 点群からの物体検出"""

    def __init__(self, model_path: str = "default_model.pth"):
        self.model = load_model(model_path)
        self._lock = threading.Lock()

    def analyze(self, lidar_data_path: str) -> ModalityResult:
        """LiDAR データを処理して ModalityResult を返す"""
        with self._lock:
            point_cloud = read_pcd(lidar_data_path)
            detections = self.model.predict(point_cloud)

            # description に検出結果をテキスト化
            description = f"LiDAR detections: {len(detections)} objects"

            return ModalityResult(
                modality_name="lidar",
                audio_cues=[],  # LiDAR は音声キューなし
                description=description,
                extra={"detections": detections},
                error=None,
            )
```

**注**: `ModalityResult` はフラット構造で返却（vision_analysis/depth_analysis 等の個別フィールドではなく）。PerceptionIR への統合は fuse_modalities で行われます。

#### Step 2: LangGraph ノードを追加

`src/safety_agent/agent.py` に追加：

```python
def lidar_node(state: AgentState, runtime: RunnableConfig) -> Dict:
    """LiDAR 分析ノード"""
    obs = state["observation"]
    analyzer = runtime.context.get("lidar_analyzer")
    
    if not analyzer:
        return {"modality_results": {}}
    
    try:
        detections = analyzer.analyze(obs.lidar_path)
        result = ModalityResult(
            modality_name="lidar",
            audio_cues=[],
            description="LiDAR detected objects",
            extra={"detections": detections},
            error=None
        )
    except Exception as e:
        result = ModalityResult(
            modality_name="lidar",
            audio_cues=[],
            description="",
            extra={},
            error=str(e)
        )
    
    return {"modality_results": {"lidar": result}}
```

#### Step 3: グラフに統合

`src/safety_agent/agent.py` の `ingest_observation()` で Send() を追加、`build_agent()` でノードを追加：

```python
# ingest_observation() 内
sends = [
    Send("vlm_node", {"observation": obs}),
    Send("audio_node", {"observation": obs}),
    Send("depth_node", {"observation": obs}),
    Send("lidar_node", {"observation": obs}),  # 新規追加
]

# build_agent() 内
graph.add_node("lidar_node", lidar_node)  # 新規ノード追加
# 自動的に join_modalities へ結果を送信（expected_modalities で制御）
```

**重要**: join_modalities は expected_modalities に基づいて動的に待機するモダリティを判定します。新しいモダリティは自動的に統合されます。

#### Step 4: Context に analyzer を追加、expected_modalities に追加

`src/run.py` の `main()` で：

```python
def main():
    # LiDAR analyzer 初期化
    lidar_analyzer = None
    if agent_cfg.get("enable_lidar", False):
        try:
            lidar_analyzer = LidarAnalyzer("models/lidar_model.pth")
        except Exception as e:
            logger.warning(f"Failed to initialize LiDAR: {e}")

    # expected_modalities に追加
    expected_modalities = ["vlm"]
    if agent_cfg.get("enable_audio", False):
        expected_modalities.append("audio")
    if agent_cfg.get("enable_depth", False):
        expected_modalities.append("depth")
    if agent_cfg.get("enable_lidar", False):
        expected_modalities.append("lidar")  # 新規追加

    # context に追加
    context = {
        "lidar_analyzer": lidar_analyzer,
        "expected_modalities": expected_modalities,
        # ... 既存の analyzers ...
    }
```

**重要**: expected_modalities は join_modalities で全モダリティの完了を待つ条件として使用されます。

### 2. 信念状態・安全判断ロジックのカスタマイズ

BeliefState + SafetyAssessment 生成ロジックを変更します。

#### カスタム BeliefState 生成

```python
def custom_update_belief_state_llm(state: AgentState, runtime: RunnableConfig) -> Dict:
    """カスタム信念状態更新ロジック"""
    ir = state.get("ir")  # 現在の PerceptionIR
    prev_belief = state.get("belief_state")  # 前フレーム BeliefState

    if not ir:
        return {"belief_state": BeliefState()}

    # カスタムロジック: vision_analysis から hazard_tracks を生成
    hazard_tracks = []
    if ir.vision_analysis and ir.vision_analysis.critical_points:
        for point in ir.vision_analysis.critical_points:
            track = HazardTrack(
                hazard_id=point.region_id,
                hazard_type="visible_hazard",
                region_id=point.region_id,
                status="new" if not prev_belief else "persistent",
                severity=point.severity,
                confidence_score=0.8,
                supporting_modalities=["vision"],
                evidence=[point.description],
            )
            hazard_tracks.append(track)

    belief_state = BeliefState(
        hazard_tracks=hazard_tracks,
        overall_risk="high" if hazard_tracks else "low",
        recommended_focus_regions=[t.region_id for t in hazard_tracks],
    )

    return {"belief_state": belief_state}
```

`agent.py` で置き換え：

```python
graph.add_node("update_belief_state_llm", custom_update_belief_state_llm)
```

#### カスタム SafetyAssessment 生成

```python
def custom_determine_next_action_llm(state: AgentState, runtime: RunnableConfig) -> Dict:
    """カスタム安全判断ロジック"""
    belief = state.get("belief_state")
    ir = state.get("ir")

    # belief_state.hazard_tracks から直接判断を生成
    if not belief or not belief.hazard_tracks:
        return {
            "assessment": SafetyAssessment(
                risk_level="low",
                safety_status="安全",
                action_type="monitor",
                reason="危険検出なし",
                priority=0.0,
            )
        }

    # hazard_tracks がある場合
    max_severity = max(
        track.severity for track in belief.hazard_tracks
    ) if belief.hazard_tracks else "low"

    risk_map = {
        "critical": "high",
        "high": "high",
        "medium": "medium",
        "low": "low",
    }
    risk_level = risk_map.get(max_severity, "low")

    assessment = SafetyAssessment(
        risk_level=risk_level,
        safety_status=f"検出された危険: {len(belief.hazard_tracks)}",
        detected_hazards=[t.hazard_type for t in belief.hazard_tracks],
        action_type="inspect_region" if belief.recommended_focus_regions else "monitor",
        target_region=belief.recommended_focus_regions[0] if belief.recommended_focus_regions else None,
        reason=f"BeliefState から {len(belief.hazard_tracks)} 件の危険トラックを検出",
        priority=min(1.0, len(belief.hazard_tracks) * 0.3),
    )

    return {"assessment": assessment}
```

`agent.py` で置き換え：

```python
graph.add_node("determine_next_action_llm", custom_determine_next_action_llm)
```

### 3. モダリティ結果の詳細化

各モダリティの出力スキーマをカスタマイズ（新しいフィールドを追加）：

```python
# src/safety_agent/schema.py のスキーマを拡張

from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class CustomVisionAnalysisResult(BaseModel):
    """拡張された VisionAnalysisResult"""
    scene_description: str
    critical_points: List[CriticalPoint] = Field(default_factory=list)
    blind_spots: List[VisionBlindSpot] = Field(default_factory=list)
    overall_risk: Literal["low", "medium", "high", "critical", "unknown"] = "unknown"
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # カスタムフィールド
    object_density: Optional[float] = None  # 画像中のオブジェクト密度（0.0-1.0）
    motion_detected: bool = False  # 動きが検出されたか
    lighting_condition: Optional[str] = None  # 照明条件（bright/normal/dark）
```

`modality_nodes.py` の VisionAnalyzer.analyze() で拡張スキーマを返却：

```python
def analyze(self, image_path: str, max_tokens: int = 4096) -> CustomVisionAnalysisResult:
    # 既存ロジック + 新フィールドの計算
    result = VisionAnalysisResult(...)

    # 拡張フィールド
    result.object_density = len(result.critical_points) / 10.0  # 正規化
    result.motion_detected = "moving" in result.scene_description.lower()

    return result
```

### 4. 設定値の追加

`configs/default.yaml` に新パラメータ追加：

```yaml
modalities:
  lidar:
    enabled: false
    model_path: "models/lidar.pth"

view_planning:
  custom_strategy: false
  coverage_weight: 0.8
  redundancy_weight: 0.2
```

`run.py` で読み込み：

```python
modality_cfg = config.get("modalities", {})
lidar_enabled = modality_cfg.get("lidar", {}).get("enabled", False)
if lidar_enabled:
    lidar_analyzer = LidarAnalyzer(
        modality_cfg["lidar"].get("model_path")
    )
```

## ベストプラクティス

### 1. スレッドセーフティ

```python
class MyAnalyzer:
    def __init__(self):
        self._lock = threading.Lock()
    
    def analyze(self, data):
        with self._lock:
            # スレッドセーフな処理
```

### 2. エラーハンドリング

```python
def my_node(state, runtime):
    try:
        result = expensive_operation()
        return {"result": result}
    except Exception as e:
        logger.warning(f"Operation failed: {e}")
        return {"errors": [str(e)]}
```

### 3. ロギング

```python
logger = setup_logger("my_module")
logger.info("Processing started")
logger.debug(f"Input: {data}")
```

### 4. テスト

```python
def test_lidar_analyzer():
    analyzer = LidarAnalyzer("models/test.pth")
    result = analyzer.analyze("data/test.pcd")

    assert isinstance(result, ModalityResult)
    assert result.modality_name == "lidar"
    assert result.description is not None
    assert result.extra.get("detections") is not None
```

## トラブルシューティング

### グラフが動作しない

```bash
python -c "
from src.safety_agent.agent import build_agent
agent = build_agent()
print(agent.get_graph().draw_mermaid())
"
```

### 新しいモダリティが反応しない

**チェックリスト**:
1. `ingest_observation()` で `Send("lidar_node", {"observation": obs})` を追加したか
2. `build_agent()` で `graph.add_node("lidar_node", lidar_node)` を追加したか
3. `run.py` の `expected_modalities` に "lidar" を追加したか（config に応じて）
4. `fuse_modalities()` で `modality_results["lidar"]` を PerceptionIR に統合したか

### join_modalities で永遠に待機

expected_modalities と received_modalities のミスマッチが原因。デバッグ:

```python
def join_modalities(state, runtime):
    expected = set(runtime.context["expected_modalities"])
    received = set(state["received_modalities"])
    logger.info(f"Expected: {expected}, Received: {received}")  # デバッグログ
    # ... 残り
```

### メモリリーク

```python
# 毎フレーム modality_results をリセット
return {"modality_results": {}}
```

## 参考資料

- [ARCHITECTURE.md](./ARCHITECTURE.md) - システムアーキテクチャ
- [QUICK_START.md](./QUICK_START.md) - セットアップガイド
- [CLAUDE.md](../CLAUDE.md) - プロジェクト仕様書
