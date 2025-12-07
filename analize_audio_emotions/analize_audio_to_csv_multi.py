import os
import csv
import time
import json
import random
import threading
import concurrent.futures
import argparse
import google.generativeai as genai
from tqdm import tqdm
from google.api_core import exceptions

# グローバル設定の読み込み
import global_config

# YAMLライブラリのインポート（利用可能な場合）
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# ==========================================
# 設定読み込みクラス
# ==========================================
class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.data = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        _, ext = os.path.splitext(self.config_path)
        ext = ext.lower()

        if ext in ['.yaml', '.yml']:
            if not YAML_AVAILABLE:
                raise ImportError("YAML config detected but 'PyYAML' is not installed. Please run `pip install PyYAML`.")
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        elif ext == '.json':
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported config file extension: {ext}")

    @property
    def list_file_path(self):
        return self.data.get("list_file_path", "")

    @property
    def audio_dir(self):
        return self.data.get("audio_dir", "")

    @property
    def output_csv(self):
        return self.data.get("output_csv", "output.csv")

    @property
    def model_name(self):
        return self.data.get("model_name", "gemini-2.5-flash")

    @property
    def max_workers(self):
        return self.data.get("max_workers", 5)

    @property
    def emotion_labels(self):
        return self.data.get("emotion_labels", [])

    @property
    def system_instruction(self):
        # JSON/YAML内の {emotion_labels} プレースホルダーを実際のリスト文字列に置換
        instruction = self.data.get("system_instruction", "")
        labels_str = ", ".join(self.emotion_labels)
        return instruction.replace("{emotion_labels}", labels_str)

# ==========================================
# メイン処理
# ==========================================

# API設定
if not global_config.API_KEY:
    raise ValueError("API_KEY is not set in global_config.py")
genai.configure(api_key=global_config.API_KEY)

# 排他制御用のロックオブジェクト（CSV書き込み時の競合防止）
csv_lock = threading.Lock()

def upload_to_gemini(path, mime_type="audio/ogg"):
    """ファイルをアップロードし、処理可能になるまで待機"""
    if not os.path.exists(path):
        return None
    
    try:
        file = genai.upload_file(path, mime_type=mime_type)
        # ファイルが処理中であれば待機
        while file.state.name == "PROCESSING":
            time.sleep(1)
            file = genai.get_file(file.name)
        
        if file.state.name == "FAILED":
            # 失敗時はクリーンアップしてエラー送出
            try:
                genai.delete_file(file.name)
            except:
                pass
            raise ValueError(f"File processing failed: {file.name}")
        return file
    except Exception as e:
        print(f"Upload failed for {path}: {e}")
        return None

def process_single_entry(entry, model, config):
    """1つのエントリを処理する関数（並列実行用）"""
    file_name = entry["filename"]
    text_content = entry["text"]
    speaker = entry["speaker"]
    file_path = os.path.join(config.audio_dir, file_name)

    # ランダムな待機時間（APIへのリクエスト集中を防ぐ）
    time.sleep(random.uniform(0.5, 2.0))

    if not os.path.exists(file_path):
        with csv_lock:
            with open(config.output_csv, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([file_name, speaker, text_content, "Error", "False", "File not found"])
        return

    uploaded_file = None
    try:
        # 1. アップロード
        ext = os.path.splitext(file_name)[1].lower()
        mime = "audio/wav" if ext == ".wav" else "audio/mp3" if ext == ".mp3" else "audio/ogg"
        
        uploaded_file = upload_to_gemini(file_path, mime_type=mime)
        if not uploaded_file:
            raise ValueError("File upload returned None")

        # 2. 推論 (リトライロジック)
        response_text = ""
        # リトライ回数設定
        max_retries = 3
        for attempt in range(max_retries):
            try:
                prompt = f"この音声のテキストは「{text_content}」です。感情、学習データとしての適性、理由を分析してください。"
                response = model.generate_content([uploaded_file, prompt])
                response_text = response.text
                break
            except exceptions.ResourceExhausted:
                # レート制限にかかった場合は長めに待機（指数バックオフ）
                wait_time = 10 * (2 ** attempt) + random.uniform(0, 5)
                time.sleep(wait_time)
            except Exception as e:
                print(f"Retry error ({file_name}): {e}")
                time.sleep(2)
        
        if not response_text:
            raise ValueError("Failed to generate content after retries")

        # 3. JSONパース
        # マークダウンのコードブロックが含まれている場合の除去処理
        cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
        result = json.loads(cleaned_text)
        
        # 4. CSVへの追記（ロックを使用して排他制御）
        with csv_lock:
            with open(config.output_csv, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    file_name,
                    speaker,
                    text_content,
                    result.get("emotion", "Unknown"),
                    result.get("is_usable", False),
                    result.get("reason", "No reason provided")
                ])

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        with csv_lock:
            with open(config.output_csv, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([file_name, speaker, text_content, "Error", "False", f"Processing Error: {e}"])

    finally:
        # クラウド上のファイルを削除（重要：ストレージ圧迫を防ぐ）
        if uploaded_file:
            try:
                genai.delete_file(uploaded_file.name)
            except:
                pass

def analyze_dataset(config_path):
    # 設定のロード
    try:
        config = ConfigLoader(config_path)
        print(f"Loaded configuration from: {config_path}")
    except Exception as e:
        print(f"Configuration Load Error: {e}")
        return

    # システムプロンプトの設定
    generation_config = {
        "response_mime_type": "application/json",
        "temperature": 0.1
    }

    model = genai.GenerativeModel(
        model_name=config.model_name,
        system_instruction=config.system_instruction,
        generation_config=generation_config
    )

    # 1. リストファイルの読み込み
    all_entries = []
    try:
        with open(config.list_file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 4:
                    all_entries.append({
                        "filename": parts[0],
                        "speaker": parts[1],
                        "text": parts[3]
                    })
    except FileNotFoundError:
        print(f"エラー: {config.list_file_path} が見つかりません。")
        return

    # 2. 再開機能：既存CSVの確認と済みデータの除外
    processed_files = set()
    if os.path.exists(config.output_csv):
        print(f"既存のCSVファイルを検出しました: {config.output_csv}")
        try:
            with open(config.output_csv, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                header = next(reader, None) # ヘッダーを読み飛ばす
                for row in reader:
                    if row: # 空行対策
                        processed_files.add(row[0]) # Filenameは1列目と仮定
        except Exception as e:
            print(f"CSV読み込み警告: {e}")

    # 未処理リストの作成
    tasks = [entry for entry in all_entries if entry["filename"] not in processed_files]
    
    print(f"全データ数: {len(all_entries)}")
    print(f"処理済み: {len(processed_files)}")
    print(f"今回処理対象: {len(tasks)}")

    if not tasks:
        print("すべてのデータが処理済みです。")
        return

    # 3. CSV初期化（ファイルが存在しない場合のみヘッダーを作成）
    if not os.path.exists(config.output_csv):
        with open(config.output_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Filename", "Speaker", "Text", "Emotion", "Is_Usable", "Reason"])

    # 4. 並列処理の実行
    print(f"Starting parallel processing with {config.max_workers} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        # タスクを投入
        future_to_file = {
            executor.submit(process_single_entry, entry, model, config): entry["filename"] 
            for entry in tasks
        }
        
        # tqdmで進捗表示
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(tasks)):
            file_name = future_to_file[future]
            try:
                future.result() # 例外があればここで発生
            except Exception as exc:
                print(f"{file_name} generated an exception: {exc}")

    print(f"完了しました。結果は {config.output_csv} を確認してください。")

if __name__ == "__main__":
    # コマンドライン引数で設定ファイルを指定可能にする
    parser = argparse.ArgumentParser(description="Audio Analysis Tool")
    # デフォルトを config.yaml に変更
    parser.add_argument("config", nargs="?", default="config.yaml", help="Path to the configuration file (JSON or YAML)")
    args = parser.parse_args()

    analyze_dataset(args.config)