# 拡張ガイド

Safety View Agent に新しいセンサーモダリティを追加する方法

## 新センサー追加ガイド（LiDAR 追加例）

本ガイドでは、LiDAR（Light Detection and Ranging）センサーを例に、fan-out アーキテクチャへ新しいモダリティを追加する全手順を説明します。

### 概要

追加に必要な変更箇所は 5 つです。

| Step | ファイル | 変更内容 |
|------|----------|---------|
| 1 | `src/safety_agent/modality_nodes.py` | 新しいアナライザークラスを追加 |
| 2 | `src/safety_agent/agent.py` | 新しいノード関数を追加 |
| 3 | `src/safety_agent/agent.py` | `ingest_observation` に `Send()` を追加 |
| 4 | `src/safety_agent/agent.py` | `build_agent()` にノードとエッジを追加 |
| 5 | テスト + 設定 | 動作確認と `configs/default.yaml` 更新 |

---

### Step 1: `modality_nodes.py` に新クラス追加

`src/safety_agent/modality_nodes.py` に `LidarAnalyzer` クラスを追加します。

```python
# ─── LidarAnalyzer ──────────────────────────────────────────────

class LidarAnalyzer:
    """LiDAR 点群データから物体を検出するアナライザー。"""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path
        # 必要に応じて点群処理モデルを初期化

    def analyze(self, lidar_path: Optional[str]) -> list[DetectedObject]:
        """LiDAR データファイルを解析して検出物体リストを返す。

        Args:
            lidar_path: 点群データファイルのパス（.pcd, .ply など）

        Returns:
            検出された物体のリスト
        """
        if not lidar_path:
            return []

        # 実装例：点群データを読み込んで物体を検出
        # ここではヒューリスティックフォールバックを示す
        objects: list[DetectedObject] = []

        try:
            # 点群ファイルの存在確認
            if not os.path.exists(lidar_path):
                return []

            # === ここに実際の点群処理ロジックを実装 ===
            # 例: Open3D, PCL, PointNet++ などを使用
            #
            # import open3d as o3d
            # pcd = o3d.io.read_point_cloud(lidar_path)
            # clusters = pcd.cluster_dbscan(eps=0.5, min_points=10)
            # for cluster_id in set(clusters):
            #     objects.append(DetectedObject(
            #         label="obstacle",
            #         confidence=0.8,
            #         bbox=BoundingBox(...)
            #     ))

        except Exception as e:
            print(f"LiDAR analysis error: {e}")

        return objects
```

**ポイント**:
- `analyze()` メソッドは `Optional[str]` パスを受け取り、`list[DetectedObject]` を返す統一インターフェース
- エラー時は空リストを返してエージェントを停止させない
- スレッドセーフティが必要な場合は `threading.Lock` を追加（`YOLODetector` を参考）

---

### Step 2: `agent.py` に新ノード追加

`src/safety_agent/agent.py` にノード関数を追加します。

```python
from .modality_nodes import (
    AudioAnalyzer,
    LidarAnalyzer,     # 追加
    ModalityResult,
    VisionAnalyzer,
    YOLODetector,
)


def lidar_node(
    state: AgentState, runtime: Runtime[ContextSchema]
) -> Dict[str, Any]:
    """LiDAR ノード：点群データから物体を検出。"""
    obs = state.get("observation")
    if obs is None:
        return {
            "modality_results": [
                ModalityResult(
                    modality_name="lidar",
                    error="No observation provided",
                )
            ]
        }

    try:
        # Observation に lidar_path フィールドを追加する必要がある
        lidar_path = getattr(obs, "lidar_path", None)
        objects = runtime.context["lidar_analyzer"].analyze(lidar_path)
        error = None
    except Exception as e:
        objects = []
        error = f"lidar_node error: {e}"

    result = ModalityResult(
        modality_name="lidar",
        objects=objects,
        error=error,
    )
    return {
        "modality_results": [result],
        "messages": [
            {
                "role": "assistant",
                "content": f"[lidar] objects={len(objects)}",
            }
        ],
    }
```

**ポイント**:
- 返す辞書のキーは `modality_results` と `messages` のみ
- `ModalityResult` の `modality_name` はユニークな文字列にする
- エラーハンドリングは `try/except` で囲み、`error` フィールドに記録

---

### Step 3: `ingest_observation` に `Send()` 追加

`ingest_observation` 関数の `sends` リストに新しい `Send` を追加します。

```python
def ingest_observation(
    state: AgentState, runtime: Runtime[ContextSchema]
) -> Command:
    # ... 既存の観測取得ロジック ...

    # fan-out: 全モダリティノードへ並列送信
    sends: list[Send] = [
        Send("vision_node", {"observation": obs}),
        Send("audio_node", {"observation": obs}),
        Send("lidar_node", {"observation": obs}),   # 追加（1行のみ）
    ]

    return Command(
        update={
            "observation": obs,
            "modality_results": [],
            "messages": [{"role": "assistant", "content": f"[ingest] fan-out -> {obs.obs_id}"}],
        },
        goto=sends,
    )
```

**この1行の追加だけで、LiDAR ノードが vision/audio と並列実行されます。**

---

### Step 4: `build_agent()` にエッジ追加

```python
def build_agent():
    builder = StateGraph(AgentState, context_schema=ContextSchema)

    # ノード登録
    builder.add_node("ingest_observation", ingest_observation)
    builder.add_node("vision_node", vision_node)
    builder.add_node("audio_node", audio_node)
    builder.add_node("lidar_node", lidar_node)        # 追加
    builder.add_node("fuse_modalities", fuse_modalities)
    builder.add_node("update_world_model", update_world_model)
    builder.add_node("propose_next_view_llm", propose_next_view_llm)
    builder.add_node("validate_and_guardrails", validate_and_guardrails)
    builder.add_node("select_view", select_view)
    builder.add_node("bump_step", bump_step)

    # エッジ設定
    builder.add_edge(START, "ingest_observation")
    builder.add_edge("vision_node", "fuse_modalities")
    builder.add_edge("audio_node", "fuse_modalities")
    builder.add_edge("lidar_node", "fuse_modalities")  # 追加
    builder.add_edge("fuse_modalities", "update_world_model")
    # ... 残りは既存のまま ...

    return builder.compile()
```

**ポイント**:
- `add_edge("lidar_node", "fuse_modalities")` を追加することで、LiDAR の完了も fan-in バリアの条件に含まれる
- 3つのモダリティ全てが完了するまで `fuse_modalities` は実行されない

---

### Step 5: テストと検証

#### 5.1 ContextSchema の更新

```python
class ContextSchema(TypedDict):
    provider: ObservationProvider
    perceiver: Perceiver
    llm: Optional[OpenAICompatLLM]
    vision_analyzer: Optional[VisionAnalyzer]
    yolo_detector: Optional[YOLODetector]
    audio_analyzer: AudioAnalyzer
    lidar_analyzer: LidarAnalyzer              # 追加
    risk_stop_threshold: float
    hazard_focus_threshold: float
```

#### 5.2 `run.py` のコンテキスト初期化

```python
context = {
    "provider": provider,
    "perceiver": perceiver,
    "llm": llm,
    "vision_analyzer": vision_analyzer,
    "yolo_detector": yolo_detector,
    "audio_analyzer": audio_analyzer,
    "lidar_analyzer": LidarAnalyzer(),          # 追加
    "risk_stop_threshold": thresholds_cfg.get("risk_stop_threshold", 0.2),
    "hazard_focus_threshold": thresholds_cfg.get("hazard_focus_threshold", 0.6),
}
```

#### 5.3 `fuse_modalities` での結果取得

既存の `fuse_modalities` ノードは `modality_results` を辞書化して処理するため、自動的に新モダリティを認識します。ただし、LiDAR 固有の結果を利用するには処理を追加します。

```python
def fuse_modalities(state, runtime):
    results = {r.modality_name: r for r in state.get("modality_results", [])}

    vision = results.get("vision")
    audio = results.get("audio")
    lidar = results.get("lidar")      # 追加

    # LiDAR の検出結果を統合
    objects = (vision.objects if vision else []) + (lidar.objects if lidar else [])
    audio_cues = audio.audio_cues if audio else []
    # ... 残りは既存のまま ...
```

#### 5.4 configs/default.yaml の更新

```yaml
modalities:
  vision:
    enabled: true
    yolo_enabled: false
  audio:
    enabled: true
  lidar:                    # 追加
    enabled: true
    model_path: null        # 点群処理モデルのパス
```

#### 5.5 テスト実行

```bash
# E2E テスト（LiDAR パスが None でもエラーにならないことを確認）
pytest tests/test_e2e.py -v

# 実際の LiDAR データで実行
python src/run.py
```

---

## 実装テンプレート

以下は新しいモダリティを追加する際のコピー＆ペースト用テンプレートです。`MODALITY_NAME` を実際の名前に置き換えてください。

### modality_nodes.py に追加

```python
class NewModalityAnalyzer:
    """新モダリティのアナライザー。"""

    def __init__(self) -> None:
        pass

    def analyze(self, data_path: Optional[str]) -> list[DetectedObject]:
        if not data_path:
            return []
        # 処理ロジックを実装
        return []
```

### agent.py に追加

```python
def new_modality_node(
    state: AgentState, runtime: Runtime[ContextSchema]
) -> Dict[str, Any]:
    """新モダリティノード。"""
    obs = state.get("observation")
    if obs is None:
        return {
            "modality_results": [
                ModalityResult(modality_name="new_modality", error="No observation")
            ]
        }

    try:
        # runtime.context から analyzer を取得して処理
        result_data = runtime.context["new_modality_analyzer"].analyze(...)
        error = None
    except Exception as e:
        result_data = []
        error = f"new_modality_node error: {e}"

    return {
        "modality_results": [
            ModalityResult(modality_name="new_modality", objects=result_data, error=error)
        ],
        "messages": [
            {"role": "assistant", "content": f"[new_modality] items={len(result_data)}"}
        ],
    }
```

### チェックリスト

- [ ] `modality_nodes.py` にアナライザークラスを追加
- [ ] `agent.py` にノード関数を追加
- [ ] `agent.py` の `ingest_observation` に `Send("new_node", ...)` を追加
- [ ] `agent.py` の `build_agent()` に `add_node` と `add_edge` を追加
- [ ] `agent.py` の `ContextSchema` に新アナライザーを追加
- [ ] `agent.py` の `fuse_modalities` で新モダリティ結果を処理
- [ ] `run.py` のコンテキスト初期化に新アナライザーを追加
- [ ] `configs/default.yaml` に新モダリティ設定を追加
- [ ] `pytest tests/ -v` が通ることを確認

---

## よくある質問

### fan-in ポイントの変更方法

デフォルトでは全モダリティノードが `fuse_modalities` に合流します。特定のモダリティを別の fan-in ポイントに向けたい場合は、エッジを変更します。

```python
# 例: LiDAR の結果を別のノードで処理してから統合
builder.add_node("lidar_preprocess", lidar_preprocess_fn)
builder.add_edge("lidar_node", "lidar_preprocess")      # LiDAR → 前処理
builder.add_edge("lidar_preprocess", "fuse_modalities")  # 前処理 → 統合
```

ただし、`fuse_modalities` は全ての入力エッジのノードが完了するまで待機するため、前処理ノードを挟むとその分だけ待機時間が追加されます。

### モダリティ結果の融合ロジック変更

`fuse_modalities` ノードの内部ロジックを変更することで、結果の融合方法をカスタマイズできます。

```python
def fuse_modalities(state, runtime):
    results = {r.modality_name: r for r in state.get("modality_results", [])}

    # カスタム融合例: 信頼度の重み付け平均
    all_objects = []
    for r in results.values():
        for obj in r.objects:
            # モダリティごとに信頼度を補正
            weight = {"vision": 1.0, "lidar": 0.9, "audio": 0.5}.get(r.modality_name, 0.5)
            adjusted = obj.model_copy(update={"confidence": obj.confidence * weight})
            all_objects.append(adjusted)

    # 重複排除（同じ label + 近い bbox）
    unique_objects = deduplicate_objects(all_objects)
    # ...
```

### ハザード推定ロジックのカスタマイズ

ハザード推定は `Perceiver._infer_hazards()` で行われます。新しいモダリティの検出結果に基づくハザードルールを追加するには、このメソッドを拡張します。

```python
# perceiver.py
def _infer_hazards(self, objects, audio):
    hazards = []

    # 既存ルール
    labels = {o.label for o in objects}
    if "person" in labels:
        hazards.append(Hazard(hazard_type="human_present", ...))

    # LiDAR 固有のルール追加
    if "obstacle" in labels:
        obstacle_objs = [o for o in objects if o.label == "obstacle"]
        if any(o.confidence > 0.9 for o in obstacle_objs):
            hazards.append(Hazard(
                hazard_type="high_confidence_obstacle",
                confidence=0.8,
                related_objects=["obstacle"],
                evidence="LiDAR detected high-confidence obstacle",
            ))

    return hazards
```

### Observation にカスタムフィールドを追加する方法

`schema.py` の `Observation` dataclass に新しいフィールドを追加します。

```python
@dataclass
class Observation:
    obs_id: str
    image_path: Optional[str] = None
    audio_text: Optional[str] = None
    lidar_path: Optional[str] = None       # 追加
    camera_pose: CameraPose = field(default_factory=CameraPose)
    video_timestamp: Optional[float] = None
```

既存のコードは `Optional` フィールドのため影響を受けません。

### モダリティを条件付きで無効化する方法

`ingest_observation` で `configs/default.yaml` の設定に基づいて `Send` リストを動的に構築できます。

```python
def ingest_observation(state, runtime):
    # ...
    sends = []

    # 設定に基づいて有効なモダリティのみ fan-out
    if runtime.context.get("vision_enabled", True):
        sends.append(Send("vision_node", {"observation": obs}))
    if runtime.context.get("audio_enabled", True):
        sends.append(Send("audio_node", {"observation": obs}))
    if runtime.context.get("lidar_enabled", False):
        sends.append(Send("lidar_node", {"observation": obs}))

    # sends が空の場合のフォールバック
    if not sends:
        sends.append(Send("audio_node", {"observation": obs}))

    return Command(update={...}, goto=sends)
```
