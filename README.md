# Voice Dataset Organizer

StyleBertVits2等の音声合成モデル学習用データセットを効率的に構築・管理するためのツールセットです。
LLMを用いた音声の感情・品質分析から、物理ファイルの整理、重複排除までを強力にサポートします。

## 主な機能

- **AI音声分析 (Audio Analysis)**
  Google Gemini API (Pro/Flash) を利用して、音声データの「感情」「テキストとの整合性」「学習データとしての適性（ノイズ有無など）」を自動判定し、CSVレポートを出力します。

- **データセットオーガナイザー (Dataset Organizer GUI)**
  分析結果CSVに基づいて、データセット内の物理ファイルの移動（感情別フォルダ分け）、不要ファイルの削除、`esd.list` の更新を一括で行います。
  - **安全設計**: 実行前に必ず変更内容をプレビューでき、自動バックアップ機能も備えています。
  - **重複検出**: ファイル名が異なっても中身が同じ音声ファイルを、波形ハッシュを用いて検出・削除できます。

- **ユーティリティ**
  TensorBoardログのCSV変換など、実験管理に役立つツールを含みます。

## セットアップ手順

### 必要要件
- Python 3.10以上推奨
- Google Gemini API Key (分析機能を使用する場合)

### インストール

1. **リポジトリのクローン**
   ```bash
   git clone https://github.com/kenafu/voice-dataset-organizer.git
   cd voice-dataset-organizer
   ```

2. **依存ライブラリのインストール**
   ```bash
   pip install -r requirements.txt
   ```
   ※ 音声再生や分析機能のために `pygame`, `librosa`, `numpy` 等がインストールされます。

3. **APIキーの設定**
   `analize_audio_emotions` ディレクトリ内の `global_config_default.py` をコピーして `global_config.py` を作成し、APIキーを設定してください。

   ```bash
   cd analize_audio_emotions
   
   # Windows (Copy)
   copy global_config_default.py global_config.py
   
   # Linux/Mac (Copy)
   cp global_config_default.py global_config.py
   ```
   
   作成した `global_config.py` をエディタで開き、`API_KEY` 変数にご自身のGoogle AI StudioのAPIキーを入力してください。

## 使い方

### 1. 音声データのAI分析

Geminiを使用して、大量の音声データを自動的に分類・評価します。

1. **設定ファイルの作成**
   分析設定を記述したYAMLファイル（例: `config.yaml`）を作成します。
   
   ```yaml
   # config.yaml (例)
   list_file_path: "F:/Dataset/MyModel/esd.list"
   audio_dir: "F:/Dataset/MyModel/raw"
   output_csv: "analysis_result.csv"
   model_name: "gemini-1.5-flash"
   max_workers: 10
   emotion_labels: ["Neutral", "Happy", "Sad", "Angry"]
   system_instruction: |
     あなたは音声分析の専門家です。
     ユーザーから提供された音声とテキストを比較し、以下のJSON形式で出力してください。
     {
       "emotion": "{emotion_labels}の中から最も近いもの",
       "is_usable": true/false (学習データとして適切か),
       "reason": "判定理由"
     }
   ```

2. **分析の実行**
   ```bash
   # リポジトリルートから実行
   python analize_audio_emotions/analize_audio_to_csv_multi.py path/to/config.yaml
   ```
   完了すると `output_csv` で指定したファイルに結果が出力されます。途中中断しても、次回実行時に続きから再開可能です。

### 2. データセットの整理 (GUI)

分析結果や手動での精査に基づいて、データセットを整理します。

```bash
python dataset_organizer/dataset_organizer.py
```

1. **基本操作**:
   - **データセットルートフォルダ**: `esd.list` が存在するディレクトリを選択します。
   - **整理分析用CSVファイル**: Step 1で出力したCSVを選択します（必須ではありません）。
   - **「データを読み込んで現状確認...」**: 現在のフォルダ構成と、CSVがある場合は変更案（移動・削除）が表示されます。
   - **「CSV整理を実行」**: プレビュー内容に基づいてファイルを操作します。変更前にバックアップが作成されます。

2. **重複音声の削除**:
   - ツールエリアの「重複音声の検出・削除」ボタンをクリックします。
   - パラメータ（デフォルト推奨）を設定して実行すると、内容が重複している音声を検出し、GUI上で聞き比べながら削除（退避）できます。

## ディレクトリ構成

- `analize_audio_emotions/`: AI分析スクリプト関連
- `dataset_organizer/`: データセット整理GUIアプリ
- `utils/`: その他便利ツール（TensorBoard変換など）

## ライセンス

MIT License
