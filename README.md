# config.yaml
list_file_path: "C:/path/to/dataset/esd.list"  # esd.listのパス
audio_dir: "C:/path/to/dataset/raw"            # 音声ファイルがあるフォルダ
output_csv: "dataset_analysis.csv"             # 出力するCSVファイル名
model_name: "gemini-2.0-flash-exp"             # 使用するモデル (flash系推奨)
max_workers: 10                                # 並列処理数

# 感情ラベルの定義（AIの分類候補）
emotion_labels:
  - "Neutral"
  - "Happy"
  - "Sad"
  - "Angry"

# AIへの指示プロンプト
system_instruction: >
  あなたは音声分析の専門家です。
  提供された音声とテキスト内容を比較し、以下のJSONフォーマットで出力してください。
  
  {
    "emotion": "感情ラベル ({emotion_labels} から選択)",
    "is_usable": true/false,  // 学習データとして適切か（ノイズ、読み間違い、音割れ等はfalse）
    "reason": "判定理由を簡潔に"
  }
