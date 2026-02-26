# チームメンバ向けドキュメント

Safety View Agent のチームメンバ向け完全ガイド

## 📚 ドキュメント一覧

### 1. **[クイックスタート](QUICK_START.md)** - 5分で始める
新規メンバが最初に読むべきガイド
- セットアップから実行まで
- 基本的なコマンド
- よくある質問（FAQ）

**所要時間:** 5-10分

### 2. **[セットアップガイド](SETUP.md)** - 詳細なセットアップ
環境構築の詳しい説明
- 前提条件の確認
- ステップバイステップ手順
- トラブル対応

**所要時間:** 15-20分

### 3. **[アーキテクチャ](ARCHITECTURE.md)** - システム設計
プロジェクトの内部構造を理解したい場合
- システム概要図
- ファイル構成と責任分担
- 実行フロー（フローチャート付き）
- LLM 互換性設計
- 拡張方法

**所要時間:** 30-45分

### 4. **[トラブルシューティング](TROUBLESHOOTING.md)** - 問題解決
エラーが出た場合に参照
- セットアップエラー
- 実行エラー
- Vision API エラー
- デバッグテクニック

**参照時間:** 必要に応じて

---

## 🚀 推奨読む順序

### 初日（新規メンバ）
1. **QUICK_START.md** を読む（5分）
2. セットアップを実行（10分）
3. テストで動作確認（2分）
4. エージェントを実行（1分）

**合計:** 20分

### 2日目以降（必要に応じて）
- **ARCHITECTURE.md** で設計を理解
- **TROUBLESHOOTING.md** で問題を解決

---

## 🎯 シーン別ガイド

### 「とりあえず動かしたい」
→ **QUICK_START.md** の「TL;DR」セクション

### 「環境構築が失敗した」
→ **SETUP.md** の「一般的な問題」セクション
→ **TROUBLESHOOTING.md** の「セットアップ関連」

### 「Vision API が空の応答を返す」
→ **TROUBLESHOOTING.md** の「Vision API 関連」

### 「実行が遅い」
→ **TROUBLESHOOTING.md** の「パフォーマンス関連」

### 「コード拡張方法を知りたい」
→ **ARCHITECTURE.md** の「拡張可能性」セクション

### 「LLM の仕組みを知りたい」
→ **ARCHITECTURE.md** の「LLM 互換性設計」セクション

---

## 📋 チェックリスト

セットアップが完了したか確認：

- [ ] `uv sync --extra dev` を実行
- [ ] `.env` ファイルを作成し API キーを設定
- [ ] `pytest tests/ -v` で全テスト合格
- [ ] `uv run python src/run.py` でエージェント実行成功
- [ ] `output/` フォルダに 3 つのファイルが生成されている

すべてチェック完了？ → **プロジェクトの使用準備完了！** 🎉

---

## 💬 よくある質問

**Q: どのファイルを編集すればいい？**
A: `src/run.py` と `src/safety_agent/` フォルダのファイルです。設定は `configs/default.yaml`。

**Q: エージェントの処理フローは？**
A: ARCHITECTURE.md の「実行フロー」セクションを参照。

**Q: 新しいハザード検出ロジックを追加するには？**
A: ARCHITECTURE.md の「拡張可能性」セクション参照。

**Q: Vision API を使わずにローカルで動かせる？**
A: はい、LLM なしでもヒューリスティックフォールバックで動作。

**Q: エージェントが遅い場合は？**
A: `max_steps` を 3 から 1-2 に削減してください。

---

## 🔧 よく使うコマンド

```bash
# 初回セットアップ
uv sync --extra dev
cp .env.example .env

# テスト実行
pytest tests/ -v

# エージェント実行
set -a && source .env && set +a && python src/run.py

# ダミー学習
python finetuning/train_dummy.py

# コードフォーマット
ruff format src/
```

---

## 📚 参考資料

### 公式ドキュメント
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Pydantic Documentation](https://docs.pydantic.dev/)

### プロジェクト内ドキュメント
- [CLAUDE.md](../CLAUDE.md) - Claude Code（AI編集ツール）向け情報

---

## 📞 サポート

問題が解決しない場合：

1. **TROUBLESHOOTING.md** で類似の問題を検索
2. GitHub Issues で同じエラーを検索
3. チームリーダーに報告（エラーログ含め）

---

**最終更新:** 2026-02-26
**対象バージョン:** Safety View Agent v1.0
**ドキュメントメンテナー:** Development Team
