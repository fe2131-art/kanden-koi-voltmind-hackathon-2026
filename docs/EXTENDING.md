# Extending Safety View Agent

このドキュメントは、現在の実装に沿って Safety View Agent を拡張するときの入口をまとめたものです。
古くなりやすい長いサンプルコードではなく、**どのファイルをどう触るか** に絞っています。

## 代表的な拡張パターン

1. 新しいモダリティを追加する
2. `BeliefState` や `SafetyAssessment` のフィールドを増やす
3. prompt を変える
4. 出力保存形式や Demo UI を変える
5. 新しい入力ソースを追加する

## 1. 新しいモダリティを追加する

最低限、次の場所を揃えます。

### 実装

- `src/safety_agent/modality_nodes.py`
  - analyzer クラスを追加
- `src/safety_agent/agent.py`
  - node 関数を追加
  - `build_agent()` にノード登録
  - `ingest_observation()` の fan-out に追加
  - `fuse_modalities()` に統合処理を追加
- `src/safety_agent/schema.py`
  - 永続化したい結果型を追加
  - 必要なら `PerceptionIR` にフィールド追加
- `src/run.py`
  - analyzer 初期化
  - `expected_modalities` へ追加
  - `context` へ analyzer / config を渡す

### 設定

- `configs/default.yaml`
  - `agent.enable_<modality>`
  - 必要なら `<modality>_every_n_frames`

### ドキュメント

- `docs/ARCHITECTURE.md`
- `docs/SETUP.md`
- `docs/TROUBLESHOOTING.md`

### テスト

- `tests/test_e2e.py`
- 必要なら専用の unit test

## 2. `BeliefState` / `SafetyAssessment` を拡張する

変更が波及する場所:

- `src/safety_agent/schema.py`
  - Pydantic モデル
- `src/safety_agent/agent.py`
  - `chat_json()` で使う schema
  - `determine_next_action_llm()` / `update_belief_state_llm()`
- `configs/prompt.yaml`
  - `belief_update`
  - `safety_assessment`
- `src/apps/server.py` / `src/apps/App.jsx`
  - Demo UI に反映する場合
- `tests/test_schema.py`
- `tests/test_llm_format.py`

### 注意

- `safety_assessment` は現在 `ActionWithGrounding` を返します
- `assessment` だけでなく `grounded_critical_points` も意識してください
- Demo UI は bbox を持つ `vision_analysis.critical_points` をオーバーレイに使っています

## 3. prompt を変える

prompt は `configs/prompt.yaml` に集約されています。

主なセクション:

- `vision_analysis`
- `depth_analysis`
- `infrared_analysis`
- `temporal_analysis`
- `audio_analysis`
- `belief_update`
- `safety_assessment`

変更時の確認ポイント:

- `schema.py` の出力構造と一致しているか
- `agent.py` の `schema_type` と一致しているか
- 日本語 / 英語のフィールドルールが downstream と矛盾しないか
- `target_region` と UI 側の region_id の整合が取れているか

## 4. 出力保存や Demo UI を変える

### 保存形式を変える

- `src/run.py`
  - `append_frame_result()`

### WebSocket メッセージを変える

- `src/apps/server.py`

### UI 表示を変える

- `src/apps/App.jsx`

### 注意

- `manifest.json` は Demo UI のポーリング基準です
- `data/perception_results/frames/*.json` の構造を変えると `server.py` も合わせて直す必要があります
- `assessment.safety_status` は TTS にも使われます

## 5. 新しい入力ソースを追加する

現在の入力モード:

- `manual`
- `inspesafe`

新しいデータセットや入力アダプタを追加する場合は:

- `src/run.py`
  - `prepare_observations_*` 関数を追加
  - `prepare_observations()` の分岐に追加
- `src/safety_agent/schema.py`
  - 必要なら `Observation` にフィールド追加

## 変更後の確認チェックリスト

- [ ] `uv run pytest tests/ -v`
- [ ] `uv run python src/run.py`
- [ ] `data/perception_results/manifest.json` が更新される
- [ ] Demo UI を使う変更なら `uv run python src/apps/server.py` + `npm run dev` でも確認
- [ ] docs と `CLAUDE.md` を更新した

## 関連ドキュメント

- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [SETUP.md](./SETUP.md)
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
- [../CLAUDE.md](../CLAUDE.md)
