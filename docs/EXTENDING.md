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

            # audio_cues に検出結果をテキスト化して格納
            description = f"LiDAR detections: {len(detections)} objects"

            return ModalityResult(
                modality_name="lidar",
                audio_cues=[],  # LiDAR は音声キューなし
                description=description,
                extra={"detections": detections},
                error=None,
            )
```

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

`src/safety_agent/agent.py` の `build_agent()` で：

```python
def build_agent() -> CompiledStateGraph:
    """Build Safety View Agent graph"""
    graph = StateGraph(AgentState)
    
    # ノード追加（fan-out/fan-in パイプラインに新規モダリティを追加）
    # 注: 既存のノード（yolo_node, vlm_node, audio_node）は ingest_observation から fan-out で並列送信される
    # 新規 lidar_node も同様に join_modalities へ参加

    graph.add_node("lidar_node", lidar_node)  # 新規

    # fan-in: join_modalities への参加（他の modality_node と同様に）
    # ingest_observation から fan-out で lidar_node へも Send() で送信するよう修正が必要
    # その後、lidar_node の出力は join_modalities へ到達
```

#### Step 4: Context に analyzer を追加

`src/run.py` で：

```python
def main():
    # LiDAR analyzer 初期化
    lidar_analyzer = None
    if agent_cfg.get("enable_lidar", False):
        try:
            lidar_analyzer = LidarAnalyzer("models/lidar_model.pth")
        except Exception as e:
            logger.warning(f"Failed to initialize LiDAR: {e}")
    
    # context に追加
    context = {
        "lidar_analyzer": lidar_analyzer,
        # ... 既存の analyzers ...
    }
```

### 2. ビュー選択ロジックのカスタマイズ

次ビュー提案のロジックを変更します。

#### カスタムスコアリング関数

```python
def custom_view_planning(state, runtime) -> Dict:
    """カスタムビュー提案ロジック"""
    world = state["world"]
    
    def score_candidate(candidate: ViewCandidate) -> float:
        unobs = world.outstanding_unobserved
        
        # カバレッジスコア
        coverage = sum(
            1 for u in unobs
            if is_visible_from(u, candidate)
        )
        
        # 冗長性ペナルティ
        redundancy = sum(
            1 for h in world.fused_hazards
            if is_visible_from(h, candidate)
        )
        
        # スコア計算
        score = coverage * 0.8 - redundancy * 0.2
        return score
    
    candidates = generate_candidates(world)
    scored = [(c, score_candidate(c)) for c in candidates]
    sorted_candidates = sorted(scored, key=lambda x: x[1], reverse=True)
    
    return {"plan": [c for c, _ in sorted_candidates[:6]]}
```

`agent.py` で置き換え：

```python
graph.add_node("propose_next_view_llm", custom_view_planning)
```

### 3. ハザード推定ロジックの拡張

機械学習ベースの分類器を追加：

```python
# src/safety_agent/hazard_classifier.py（新規）

class HazardClassifier:
    """学習済みモデルでハザード分類"""
    
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
    
    def classify(self, features: dict) -> tuple[str, float]:
        """ハザード分類"""
        feature_vector = [
            features.get("object_count", 0),
            features.get("avg_distance", 100),
            features.get("motion_magnitude", 0),
        ]
        
        prediction = self.model.predict([feature_vector])[0]
        confidence = self.model.predict_proba([feature_vector])[0].max()
        
        return prediction, confidence
```

`perceiver.py` で使用：

```python
class Perceiver:
    def __init__(self, hazard_classifier_path: str = None):
        self.classifier = None
        if hazard_classifier_path:
            self.classifier = HazardClassifier(hazard_classifier_path)
    
    def _infer_hazards(self, objects, audio_cues) -> list[Hazard]:
        """ハザード推定"""
        hazards = []
        
        for obj in objects:
            if obj.class_name in ["vehicle", "person"]:
                if self.classifier:
                    features = {
                        "object_count": len(objects),
                        "avg_distance": np.mean([o.distance for o in objects]),
                    }
                    hazard_type, conf = self.classifier.classify(features)
                    if conf > 0.7:
                        hazards.append(Hazard(
                            class_name=hazard_type,
                            confidence=conf,
                            bbox=obj.bbox
                        ))
        
        return hazards
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

### ノードがスキップされる

`add_node()` と `add_edge()` の呼び出し順序を確認してください。

### メモリリーク

```python
# 毎フレーム modality_results をリセット
return {"modality_results": {}}
```

## 参考資料

- [ARCHITECTURE.md](./ARCHITECTURE.md) - システムアーキテクチャ
- [QUICK_START.md](./QUICK_START.md) - セットアップガイド
- [CLAUDE.md](../CLAUDE.md) - プロジェクト仕様書
