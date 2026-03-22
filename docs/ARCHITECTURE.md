# Safety View Agent Architecture

Safety View Agent は、LangGraph ベースのマルチモーダル安全監視パイプラインです。
RGB、音声、深度、赤外線、時系列差分、SAM3 セグメンテーションを統合し、`BeliefState` と `ActionWithGrounding` を生成します。

## 全体像

```text
Observation
  ↓
ingest_observation
  ↓ fan-out
vlm_node / audio_node / depth_node / infrared_node / temporal_node / sam3_node
  ↓ fan-in
join_modalities
  ↓
fuse_modalities
  ↓
update_belief_state_llm
  ↓
determine_next_action_llm
  ↓
emit_output
  ↓
append_frame_result (run.py)
  ├─ data/perception_results/
  ├─ data/voice/           (TTS)
  └─ data/flow.md          (Mermaid)
```

## 主要ファイル

- `src/run.py`
  - 設定と prompt 読み込み
  - 入力データ準備
  - 各 analyzer 初期化
  - `append_frame_result()` による結果保存
  - TTS 実行
- `src/safety_agent/agent.py`
  - LangGraph 定義
  - LLM クライアント
  - fan-out / fan-in と reducer
  - `BeliefState` / `ActionWithGrounding` 生成
- `src/safety_agent/modality_nodes.py`
  - Vision / Audio / Depth / Infrared / Temporal / SAM3 analyzer
- `src/safety_agent/schema.py`
  - `PerceptionIR`
  - `BeliefState`
  - `SafetyAssessment`
  - `ActionWithGrounding`
- `src/apps/server.py`
  - `data/perception_results/manifest.json` を監視
  - WebSocket で Demo UI へストリーミング
- `src/apps/App.jsx`
  - 動画再生
  - BBox オーバーレイ
  - Assessment / Depth / Infrared / Audio cue / Voice 表示

## 入力モード

### `manual`

- `data/videos/` の動画を使うか
- `data/frames/` の既存フレームを使います

動画がある場合は、`run.py` が lazy frame extraction を使ってフレーム抽出と推論をオーバーラップさせます。

### `inspesafe`

- `data.inspesafe.dataset_path`
- `data.inspesafe.session`

をもとに `DATA_PATH/<session>` を解決し、以下を自動で行います。

- `*_visible_*.mp4` → `data/frames/`
- `*_infrared_*.mp4` → `data/infrared_frames/`
- `*_audio_*.wav` → `data/audio/audio.wav`
- デモ用動画 → `data/videos/video.mp4`

## グラフ内の主要状態

### `Observation`

1 フレーム分の入力を表す軽量データです。

- `image_path`
- `prev_image_path`
- `audio_path`
- `infrared_image_path`
- `video_timestamp`

### `PerceptionIR`

各モダリティの結果を束ねた、そのフレームの知覚統合結果です。

- `vision_analysis`
- `audio`
- `depth_analysis`
- `infrared_analysis`
- `temporal_analysis`
- `sam3_analysis`
- `provisional_points`

### `BeliefState`

フレーム間で引き継ぐ危険状態です。

- `hazard_tracks`
- `overall_risk`
- `recommended_focus_regions`

### `ActionWithGrounding`

最終判断ノードの出力です。

- `assessment: SafetyAssessment`
- `grounded_critical_points`

`assessment` だけでなく、SAM3 region と紐づいた最終危険点も同時に返します。

## fan-out / fan-in

`ingest_observation()` は有効モダリティに応じて `Send()` を発行します。
待機対象は `run.py` で組み立てる `expected_modalities` に依存します。

例:

- 常に `vlm`
- `agent.enable_audio=true` なら `audio`
- `agent.enable_depth=true` なら `depth`
- `agent.enable_infrared=true` なら `infrared`
- `agent.enable_temporal=true` なら `temporal`
- `agent.enable_sam3=true` なら `sam3`

`join_modalities()` は、これらが揃うまで待ち、同一フレームでの二重 `fuse` を `barrier_obs_id` で防ぎます。

## 最終判断

### `update_belief_state_llm`

入力:

- `PerceptionIR`
- 既存の `belief_state`

出力:

- 新しい `BeliefState`

ここでは継続・悪化・改善・解消を管理します。

### `determine_next_action_llm`

入力:

- `PerceptionIR`
- `BeliefState`
- 任意で `assessment_history`

出力:

- `ActionWithGrounding`

このノードが `configs/prompt.yaml` の `safety_assessment` を使い、`assessment` と `grounded_critical_points` を同時生成します。

## Structured Outputs

### vLLM

- `response_format: json_schema`
- スキーマは `schema.py` の Pydantic モデルから生成

### OpenAI

- `response_format: json_object`
- 追加で prompt から JSON 構造を誘導

## 出力保存

`run.py` の `append_frame_result()` は、1 フレームごとに以下を保存します。

```text
data/perception_results/
├── manifest.json
└── frames/
    ├── 000000_img_0.json
    ├── 000001_img_1.json
    └── ...
```

新しい実行を始めると、既存の `data/perception_results/` は `data/results_archive/<timestamp>/` に移動されます。

## Demo UI との接続

`src/apps/server.py` は `manifest.json` をポーリングし、新規フレームだけを WebSocket で送ります。

UI が使う主なデータは以下です。

- `assessment`
- `vision_analysis.scene_description`
- `vision_analysis.critical_points` のうち `normalized_bbox` を持つもの
- `audio`
- `data/depth/` の画像
- `data/infrared_frames/` の画像
- `data/voice/` の WAV

### 重要な補足

- Demo UI の BBox オーバーレイは `vision_analysis.critical_points` ベースです
- `assessment.target_region` がハイライトに使われるため、region_id の整合性が重要です
- `assessment.safety_status` は TTS の入力でもあります

## 設定上の注意

- `agent.enable_sam3` が実行スイッチです
- `sam3:` セクションは analyzer のしきい値・prompt・mask 保存設定です
- `sam3.checkpoint_path` の既定値はチーム環境のローカルパスなので、そのままでは他環境で動かないことがあります
- `configs/default.yaml` の既定値は `inspesafe + vllm` 前提です

## 関連ドキュメント

- [SETUP.md](./SETUP.md)
- [QUICK_START.md](./QUICK_START.md)
- [DEMO_APP.md](./DEMO_APP.md)
- [INSPESAFE_INTEGRATION.md](./INSPESAFE_INTEGRATION.md)
- [EXTENDING.md](./EXTENDING.md)
