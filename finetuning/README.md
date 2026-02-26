# ファインチューニング

Safety View Agent モデル適応用ファインチューニングモジュール。

## 概要

提供機能：
- **ダミー学習**: CPU専用、~1秒で完了、テスト/CI用
- **LoRA ファインチューニング**: 本格学習（Low-Rank Adaptation）（PyTorch必要）
- **データセットユーティリティ**: JSONL検証、Alpacaフォーマット変換
- **モデルエクスポート**: vLLM等デプロイ用準備

## ダミー学習（CI対応）

### クイックスタート

```bash
uv run python finetuning/scripts/train_dummy.py
```

`finetuning/outputs/dummy_run/` に出力アーティファクト作成：
- `adapter_config.json` - LoRA設定
- `adapter_model.safetensors` - ダミーモデル重み
- `adapter_model.json` - モデルメタデータ
- `training_summary.json` - 学習統計

**依存関係不要** - CPU実行、<1秒で完了。

## 本格的LoRAファインチューニング

### 必須環境

```bash
uv sync --extra finetune
```

インストール: torch, transformers, peft, datasets, safetensors

### 設定

`finetuning/configs/lora.yaml` を編集：
```yaml
training:
  epochs: 3
  batch_size: 4
  learning_rate: 0.0001

model:
  base_model: "meta-llama/Llama-2-7b"  # または他のモデル
  lora_r: 8
  lora_alpha: 16
```

### 学習

```bash
uv run python finetuning/scripts/train_lora.py \
    --model meta-llama/Llama-2-7b \
    --dataset finetuning/data/samples/dummy_instructions.jsonl \
    --output-dir finetuning/outputs/lora_run \
    --epochs 3 \
    --batch-size 4
```

## データセット準備

### フォーマット (JSONL)

各行は次の構造のJSONオブジェクト：
```json
{
  "instruction": "タスク説明",
  "input": "オプション: コンテキスト",
  "output": "期待される応答"
}
```

例：
```json
{"instruction": "あなたは安全支援エージェントです。...", "input": "{...}", "output": "{...}"}
```

### 検証

```bash
uv run python finetuning/scripts/prepare_dataset.py \
    --output finetuning/data/custom/dataset.jsonl \
    --create-dummy
```

JSONLフォーマット検証とサマリー表示。

## モデルエクスポート

### vLLM用

```bash
uv run python finetuning/scripts/export_for_vllm.py \
    --adapter finetuning/outputs/lora_run \
    --base-model meta-llama/Llama-2-7b \
    --output finetuning/outputs/vllm_ready
```

アダプタ重みをベースモデルとマージしてvLLMデプロイ対応。

## サポート対象ベースモデル

- `meta-llama/Llama-2-7b` (デフォルト)
- `meta-llama/Llama-2-13b`
- `mistralai/Mistral-7B-v0.1`
- HuggingFace アテンション層搭載のあらゆるモデル

## 学習データ要件

### 最小要件
- 3+ サンプル（テスト用）
- 有効なJSON Lines フォーマット
- 一貫した instruction/output 構造

### 推奨
- 100+ サンプル
- 多様なタスク種別
- バランス取れた難度
- クリーン・検証済みデータ

## LoRA パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|---------|------|
| `lora_r` | 8 | LoRA ランク |
| `lora_alpha` | 16 | スケーリング係数 |
| `lora_dropout` | 0.1 | 正則化用ドロップアウト |
| `target_modules` | ["q_proj", "v_proj"] | 適応レイヤ |

小さい値 (r=4-8) で素早い実験、大きい値 (r=32-64) で高品質。

## トラブルシューティング

### メモリ不足

batch_size を減らすか gradient_accumulation_steps 使用：
```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 2
```

### 学習が遅い

- エポック削減 (テスト用1-2)
- max_samples 削減
- batch_size 削減（学習との トレードオフ）

### 無効なモデル名

HuggingFace Hub に存在確認：
```bash
huggingface-cli model-info meta-llama/Llama-2-7b
```

## CI 統合

ダミーファインチューニング GPU なしで CI 実行：

```bash
uv run pytest tests/integration/test_finetune_dummy.py -v
```

検証項目：
- ✅ 学習エラー無く完了
- ✅ 出力ディレクトリ作成
- ✅ すべてのアーティファクト存在
- ✅ サマリー JSON 有効

## 高度な使用法

### カスタムプロンプトテンプレート

`finetuning/src/templates.py` を編集：
```python
def custom_template(instruction, input_text, output):
    return f"<custom>{instruction}...</custom>"
```

### 学習再開

```bash
# 既存チェックポイント指定
uv run python finetuning/scripts/train_lora.py \
    --model meta-llama/Llama-2-7b \
    --resume-from finetuning/outputs/checkpoint-100
```

### マルチGPU学習

```bash
# 利用可能GPU自動使用
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python finetuning/scripts/train_lora.py ...
```

## 参考資料

- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)
- [LoRA論文](https://arxiv.org/abs/2106.09685)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
