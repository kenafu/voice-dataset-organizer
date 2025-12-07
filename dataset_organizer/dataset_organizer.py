import os
import csv
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
from datetime import datetime
import threading
import hashlib
from collections import defaultdict

# --- オプショナルなライブラリのインポート ---

# 音声再生用 (pygame)
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

# 音声解析用 (librosa, numpy) - 重複検出に使用
try:
    import librosa
    import numpy as np
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


# --- 音声解析用関数 (dedup_esd.pyより移植) ---
def audio_content_hash(path, sr=16000, top_db=40, hash_sr=2000, q_levels=15):
    """
    録音開始/終了の無音のブレを吸収しつつ、
    内容が「ほぼ同じ」なら同一ハッシュになるようにした音声ハッシュ。
    """
    if not HAS_LIBROSA:
        return None

    try:
        # 音声読み込み（モノラル・指定 sr にリサンプル）
        y, _ = librosa.load(path, sr=sr, mono=True)

        # 頭と尻の無音を削る
        y, _ = librosa.effects.trim(y, top_db=top_db)

        if len(y) == 0:
            return None

        # 振幅正規化
        max_abs = np.max(np.abs(y))
        if max_abs == 0:
            return None
        y = y / max_abs

        # 判定用にさらに低いサンプリングレートへダウンサンプル
        if sr != hash_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=hash_sr)

        # かなり荒く量子化
        half = (q_levels - 1) / 2.0
        y_q = np.round(y * half).astype(np.int8)

        # バイト列にしてハッシュ
        m = hashlib.sha1()
        m.update(y_q.tobytes())
        return m.hexdigest()
    except Exception as e:
        print(f"Hash calculation failed for {path}: {e}")
        return None


class DatasetOrganizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("StyleBertVits2 Dataset Organizer & Dedup Tool")
        self.root.geometry("1100x850") # 少し高さを増やす

        # 変数
        self.dataset_dir = tk.StringVar()
        self.csv_path = tk.StringVar()
        
        # 重複検出パラメータ用変数
        self.dedup_top_db = tk.IntVar(value=40)
        self.dedup_hash_sr = tk.IntVar(value=2000)
        self.dedup_q_levels = tk.IntVar(value=15)
        
        # 内部データ保持用
        self.preview_data = {} 
        self.raw_files_map = {} 
        self.dedup_results = [] # 重複検出結果保持用
        self.dedup_data_map = {} # TreeviewのIDとデータの紐付け用

        # Pygame初期化
        if HAS_PYGAME:
            try:
                pygame.mixer.init()
            except Exception as e:
                print(f"Pygame init failed: {e}")

        # UI構築
        self.create_widgets()

    def create_widgets(self):
        # --- 上部：設定エリア ---
        frame_top = tk.Frame(self.root)
        frame_top.pack(fill="x", padx=10, pady=5)

        frame_dir = tk.LabelFrame(frame_top, text="基本設定", padx=10, pady=10)
        frame_dir.pack(fill="x")

        # データセットディレクトリ選択
        lbl_dir = tk.Label(frame_dir, text="データセットルートフォルダ (esd.listがある場所):")
        lbl_dir.grid(row=0, column=0, sticky="w")
        
        frame_input_dir = tk.Frame(frame_dir)
        frame_input_dir.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        entry_dir = tk.Entry(frame_input_dir, textvariable=self.dataset_dir, width=60)
        entry_dir.pack(side="left", fill="x", expand=True)
        btn_dir = tk.Button(frame_input_dir, text="参照", command=self.select_directory)
        btn_dir.pack(side="right", padx=5)

        # CSVファイル選択
        lbl_csv = tk.Label(frame_dir, text="整理分析用CSVファイル (任意 / dataset_analysis.csv / 差分のみでも可):")
        lbl_csv.grid(row=2, column=0, sticky="w")
        
        frame_input_csv = tk.Frame(frame_dir)
        frame_input_csv.grid(row=3, column=0, sticky="ew", pady=(0, 5))
        entry_csv = tk.Entry(frame_input_csv, textvariable=self.csv_path, width=60)
        entry_csv.pack(side="left", fill="x", expand=True)
        btn_csv = tk.Button(frame_input_csv, text="参照", command=self.select_csv)
        btn_csv.pack(side="right", padx=5)
        
        frame_dir.columnconfigure(0, weight=1)

        # --- 中部：ツールボタンエリア ---
        frame_tools = tk.LabelFrame(self.root, text="ツール", padx=10, pady=5)
        frame_tools.pack(fill="x", padx=10, pady=5)

        # 左側：実行ボタン
        frame_tools_left = tk.Frame(frame_tools)
        frame_tools_left.pack(side="left", padx=5, fill="y")
        
        btn_dedup = tk.Button(
            frame_tools_left, 
            text="重複音声の検出・削除 (Experimental)", 
            command=self.start_dedup_check,
            bg="#fff9c4",
            height=2
        )
        btn_dedup.pack(side="top", pady=5)
        
        if not HAS_LIBROSA:
            btn_dedup.config(state="disabled", text="重複チェック不可 (librosa不足)")

        lbl_tool_info = tk.Label(frame_tools_left, text="※CSV不要", font=("Arial", 8))
        lbl_tool_info.pack(side="top")

        # 右側：パラメータ設定
        frame_params = tk.LabelFrame(frame_tools, text="重複判定パラメータ (検出漏れがある場合は値を小さく)", padx=5, pady=5)
        frame_params.pack(side="left", padx=20, fill="both", expand=True)

        # Top DB
        tk.Label(frame_params, text="無音閾値(dB):").grid(row=0, column=0, sticky="e")
        tk.Spinbox(frame_params, from_=10, to=80, textvariable=self.dedup_top_db, width=5).grid(row=0, column=1, sticky="w", padx=5)
        tk.Label(frame_params, text="(デフォルト: 40)").grid(row=0, column=2, sticky="w")

        # Hash SR
        tk.Label(frame_params, text="比較用レート(Hz):").grid(row=1, column=0, sticky="e")
        tk.Spinbox(frame_params, from_=500, to=16000, increment=100, textvariable=self.dedup_hash_sr, width=5).grid(row=1, column=1, sticky="w", padx=5)
        tk.Label(frame_params, text="(デフォルト: 2000, 下げると曖昧に)").grid(row=1, column=2, sticky="w")

        # Q Levels
        tk.Label(frame_params, text="量子化レベル:").grid(row=2, column=0, sticky="e")
        tk.Spinbox(frame_params, from_=2, to=100, textvariable=self.dedup_q_levels, width=5).grid(row=2, column=1, sticky="w", padx=5)
        tk.Label(frame_params, text="(デフォルト: 15, 下げると曖昧に)").grid(row=2, column=2, sticky="w")


        # --- 中部：プレビュー操作 ---
        frame_action = tk.Frame(self.root, pady=5)
        frame_action.pack(fill="x", padx=10)
        
        btn_load = tk.Button(frame_action, text="データを読み込んで現状確認 / 差分プレビュー表示", command=self.load_preview, bg="#e0f7fa", font=("Arial", 10, "bold"), height=1)
        btn_load.pack(side="left", fill="x", expand=True, padx=(0, 5))

        # --- メイン：タブエリア（Notebook） ---
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)
        
        # 初期タブ（説明用）
        self.tab_info = tk.Frame(self.notebook)
        self.notebook.add(self.tab_info, text="Info")
        
        info_text = (
            "【機能1: CSVベースの整理・現状確認】\n"
            "「データを読み込んで現状確認 / 差分プレビュー表示」ボタンを押すと、\n"
            "現在のフォルダ構成とesd.listの内容が表示されます。\n"
            "CSVファイルが指定されている場合は、整理（移動・削除）の構成案が表示されます。\n\n"
            "【機能2: 重複音声の検出】\n"
            "「重複音声の検出・削除」ボタンを押すと、音声の内容を解析し、\n"
            "まったく同じ音声データ（重複）を見つけ出します。\n"
            "※この機能にはCSVファイルは不要です。\n\n"
            "＜重複判定のコツ＞\n"
            "検出されない重複がある場合は、パラメータを調整してください。\n"
            "・量子化レベルを「10」などに下げる -> 音量の細かい違いを無視します\n"
            "・比較用レートを「1000」などに下げる -> タイミングのズレを許容します"
        )
        lbl_info = tk.Label(self.tab_info, text=info_text, padx=20, pady=20, justify="left")
        lbl_info.pack()

        # --- 下部：実行とログ ---
        frame_bottom = tk.Frame(self.root)
        frame_bottom.pack(fill="x", padx=10, pady=10)

        # 整理実行ボタンはCSV読み込み時のみ有効なイメージだが、常時表示しておく
        btn_run = tk.Button(frame_bottom, text="CSV整理を実行 (バックアップ作成 -> 移動・削除)", command=self.run_processing, bg="#ffcdd2", font=("Arial", 10, "bold"), height=2)
        btn_run.pack(fill="x", pady=(0, 10))

        lbl_log = tk.Label(frame_bottom, text="実行ログ:")
        lbl_log.pack(anchor="w")
        self.log_area = scrolledtext.ScrolledText(frame_bottom, state='disabled', height=6)
        self.log_area.pack(fill="both", expand=True)

        self.check_libraries()

    def check_libraries(self):
        msgs = []
        if not HAS_PYGAME:
            msgs.append("[情報] pygameがありません。音声再生機能は使用できません。('pip install pygame')")
        if not HAS_LIBROSA:
            msgs.append("[情報] librosa/numpyがありません。重複検出機能は使用できません。('pip install librosa numpy')")
        
        if msgs:
            self.log("\n".join(msgs))

    def log(self, message):
        self.log_area.configure(state='normal')
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.configure(state='disabled')
        self.root.update()

    def select_directory(self):
        path = filedialog.askdirectory()
        if path:
            self.dataset_dir.set(path)

    def select_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if path:
            self.csv_path.set(path)

    # ---------------------------------------------------------
    # CSVベースの整理機能 (Dataset Organizer)
    # ---------------------------------------------------------
    def load_preview(self):
        """CSVとesd.listを読み込み、差分を考慮してGUI上のタブに展開する"""
        dataset_root = self.dataset_dir.get()
        csv_file = self.csv_path.get()

        if not dataset_root or not os.path.exists(dataset_root):
            messagebox.showerror("エラー", "有効なデータセットフォルダを指定してください。")
            return
        
        # CSVは任意（なければ現状確認モード）
        # if not csv_file or not os.path.exists(csv_file):
        #     messagebox.showerror("エラー", "有効なCSVファイルを指定してください。")
        #     return

        esd_path = os.path.join(dataset_root, "esd.list")
        raw_dir = os.path.join(dataset_root, "raw")

        if not os.path.exists(esd_path) or not os.path.exists(raw_dir):
            messagebox.showerror("エラー", "esd.list または rawフォルダが見つかりません。")
            return

        self.log("データを読み込んでいます...")

        # 1. 現在のrawファイル一覧を取得（再生用パス解決のため）
        self.raw_files_map = {}
        for root, dirs, files in os.walk(raw_dir):
            for file in files:
                if file.endswith(('.wav', '.ogg', '.mp3')):
                    self.raw_files_map[file] = os.path.join(root, file)

        # 2. CSV読み込み (指定がある場合のみ)
        csv_data = {}
        if csv_file and os.path.exists(csv_file):
            try:
                with open(csv_file, 'r', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    reader.fieldnames = [name.strip() for name in reader.fieldnames]
                    for row in reader:
                        filename = row.get('Filename', '').strip()
                        filename = os.path.basename(filename)
                        if filename:
                            csv_data[filename] = row
            except Exception as e:
                messagebox.showerror("エラー", f"CSV読み込みエラー: {e}")
                return
        elif csv_file:
            # パスが入っているがファイルがない場合
            messagebox.showerror("エラー", "指定されたCSVファイルが見つかりません。")
            return
        else:
            self.log("CSVファイルが指定されていないため、現状の構成を表示します。")

        # 3. esd.list読み込みとデータ結合 (差分ロジック)
        self.preview_data = {} # {Category_Name: [item_dict, ...]}
        
        stats = {"moved": 0, "deleted": 0, "kept": 0}

        try:
            with open(esd_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split('|')
                if len(parts) < 2: continue
                
                current_full_path = parts[0]
                filename = os.path.basename(current_full_path)
                
                current_dir = os.path.dirname(current_full_path)
                current_dir = current_dir.replace('\\', '/').strip('/')
                if not current_dir or current_dir == '.':
                    current_emotion = "Unclassified"
                else:
                    current_emotion = current_dir
                
                text = parts[3] if len(parts) > 3 else ""
                
                if filename in csv_data:
                    row = csv_data[filename]
                    is_usable = row.get('Is_Usable', 'False').strip().lower() == 'true'
                    new_emotion = row.get('Emotion', 'Neutral').strip()
                    reason = row.get('Reason', '').strip()
                    
                    new_emotion = "".join(c for c in new_emotion if c.isalnum() or c in (' ', '_', '-')).strip()
                    if not new_emotion: new_emotion = "Neutral"

                    if not is_usable:
                        category = "削除対象(Excluded)"
                        status = "Delete"
                        stats["deleted"] += 1
                    else:
                        category = new_emotion
                        if current_emotion != new_emotion and current_emotion != "Unclassified":
                            status = f"Move ({current_emotion} -> {new_emotion})"
                            stats["moved"] += 1
                        elif current_emotion == "Unclassified":
                            status = f"New Sort ({new_emotion})"
                            stats["moved"] += 1
                        else:
                            status = "Keep (Verified)"
                            stats["kept"] += 1

                else:
                    # CSVにない、またはCSV未使用時
                    category = current_emotion if current_emotion != "Unclassified" else "Unclassified (Root)"
                    reason = "-"
                    status = "Keep (No CSV Entry)"
                    stats["kept"] += 1

                if category not in self.preview_data:
                    self.preview_data[category] = []
                
                self.preview_data[category].append({
                    "filename": filename,
                    "text": text,
                    "reason": reason,
                    "status": status,
                    "path": self.raw_files_map.get(filename, ""),
                    "original_line": line.strip()
                })

            self.update_notebook_ui()
            if csv_data:
                self.log(f"差分解析完了: 移動・変更 {stats['moved']}件, 削除予定 {stats['deleted']}件, 維持 {stats['kept']}件")
            else:
                self.log(f"現状確認完了: {stats['kept']}件のデータを読み込みました。")

        except Exception as e:
            messagebox.showerror("エラー", f"データ解析エラー: {e}")
            self.log(f"エラー詳細: {e}")

    def update_notebook_ui(self):
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)
        
        categories = sorted(self.preview_data.keys())
        sorted_cats = [c for c in categories if "Excluded" not in c] + [c for c in categories if "Excluded" in c]

        for cat in sorted_cats:
            items = self.preview_data[cat]
            tab_frame = tk.Frame(self.notebook)
            self.notebook.add(tab_frame, text=f"{cat} ({len(items)})")
            
            frame_tab_top = tk.Frame(tab_frame, pady=5)
            frame_tab_top.pack(fill="x", padx=5)
            
            btn_export = tk.Button(
                frame_tab_top, 
                text=f"「{cat}」のリストをエクスポート", 
                command=lambda c=cat, i=items: self.export_category_list(c, i)
            )
            btn_export.pack(side="left")

            cols = ("Status", "Filename", "Text", "Reason")
            tree = ttk.Treeview(tab_frame, columns=cols, show='headings')
            
            tree.heading("Status", text="Status (Change)")
            tree.column("Status", width=180)
            tree.heading("Filename", text="Filename")
            tree.column("Filename", width=150)
            tree.heading("Text", text="Text")
            tree.column("Text", width=400)
            tree.heading("Reason", text="Reason")
            tree.column("Reason", width=300)

            tree.tag_configure("move", foreground="blue")
            tree.tag_configure("delete", foreground="red")
            tree.tag_configure("keep", foreground="black")

            vsb = ttk.Scrollbar(tab_frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=vsb.set)
            
            tree.pack(side="left", fill="both", expand=True)
            vsb.pack(side="right", fill="y")

            for item in items:
                status_text = item["status"]
                tag = "keep"
                if "Move" in status_text or "New" in status_text:
                    tag = "move"
                elif "Delete" in status_text:
                    tag = "delete"
                
                tree.insert("", "end", values=(status_text, item["filename"], item["text"], item["reason"]), tags=(tag,))
            
            tree.bind("<<TreeviewSelect>>", self.on_tree_select)

    def export_category_list(self, category, items):
        if not items: return

        safe_cat = "".join(c for c in category if c.isalnum() or c in (' ', '_', '-')).strip()
        default_filename = f"esd_{safe_cat}.list"
        
        save_path = filedialog.asksaveasfilename(
            defaultextension=".list",
            initialfile=default_filename,
            filetypes=[("List Files", "*.list"), ("All Files", "*.*")],
            title=f"「{category}」のリストを保存"
        )

        if save_path:
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    for item in items:
                        line = item.get("original_line", "").strip()
                        if line:
                            f.write(line + "\n")
                messagebox.showinfo("完了", f"リストを保存しました。\n{save_path}")
            except Exception as e:
                messagebox.showerror("エラー", f"保存に失敗しました:\n{e}")

    def on_tree_select(self, event):
        tree = event.widget
        selection = tree.selection()
        if not selection: return
        item_id = selection[0]
        values = tree.item(item_id, 'values')
        filename = values[1]
        file_path = self.raw_files_map.get(filename)
        if file_path and os.path.exists(file_path):
            self.play_audio(file_path)

    def play_audio(self, file_path):
        if not HAS_PYGAME: return
        def _play():
            try:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.play()
            except Exception as e:
                print(e)
        threading.Thread(target=_play, daemon=True).start()

    def run_processing(self):
        """CSVベースの整理実行"""
        dataset_root = self.dataset_dir.get()
        csv_file = self.csv_path.get()
        
        if not dataset_root or not csv_file:
            messagebox.showwarning("注意", "先にフォルダとCSVを選択してください。")
            return

        if not messagebox.askyesno("確認", "プレビューの内容に基づいてデータセットを整理しますか？\n\n・バックアップ作成\n・不要ファイル削除\n・感情フォルダへの移動\n・esd.list更新"):
            return

        try:
            self.log("=== 整理処理開始 ===")
            
            esd_path = os.path.join(dataset_root, "esd.list")
            raw_dir = os.path.join(dataset_root, "raw")

            # バックアップ
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_root = os.path.join(dataset_root, f"backup_{timestamp}")
            self.log(f"バックアップを作成中...: {backup_root}")
            os.makedirs(backup_root, exist_ok=True)
            if os.path.exists(esd_path):
                shutil.copy2(esd_path, os.path.join(backup_root, "esd.list"))
            if os.path.exists(raw_dir):
                shutil.copytree(raw_dir, os.path.join(backup_root, "raw"))
            
            # CSV読み込み
            csv_data = {}
            with open(csv_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                reader.fieldnames = [name.strip() for name in reader.fieldnames]
                for row in reader:
                    fn = row.get('Filename', '').strip()
                    fn = os.path.basename(fn)
                    if fn: csv_data[fn] = row
            
            new_esd_lines = []
            files_to_delete = []
            files_to_move = []

            with open(esd_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            current_files = {}
            for root, dirs, files in os.walk(raw_dir):
                for file in files:
                    if file.endswith(('.wav', '.ogg', '.mp3')):
                        current_files[file] = os.path.join(root, file)

            moved_count = 0
            deleted_count = 0

            for line in lines:
                line = line.strip()
                if not line: continue
                parts = line.split('|')
                if len(parts) < 2: continue
                
                current_full_path = parts[0]
                filename = os.path.basename(current_full_path)
                
                if filename in csv_data:
                    row = csv_data[filename]
                    is_usable = row.get('Is_Usable', 'False').strip().lower() == 'true'
                    emotion = row.get('Emotion', 'Neutral').strip()
                    emotion = "".join(c for c in emotion if c.isalnum() or c in (' ', '_', '-')).strip()
                    if not emotion: emotion = "Neutral"

                    file_abs_path = current_files.get(filename)

                    if not is_usable:
                        deleted_count += 1
                        if file_abs_path: files_to_delete.append(file_abs_path)
                    else:
                        new_rel_path = f"{emotion}/{filename}"
                        parts[0] = new_rel_path
                        new_esd_lines.append("|".join(parts) + "\n")

                        if file_abs_path:
                            dest_dir = os.path.join(raw_dir, emotion)
                            dest_path = os.path.join(dest_dir, filename)
                            if os.path.normpath(file_abs_path) != os.path.normpath(dest_path):
                                files_to_move.append((file_abs_path, dest_dir, dest_path))
                                moved_count += 1
                else:
                    new_esd_lines.append(line + "\n")

            # ファイル操作
            for f_path in files_to_delete:
                if os.path.exists(f_path): os.remove(f_path)
            
            for src, dest_d, dest_f in files_to_move:
                os.makedirs(dest_d, exist_ok=True)
                if os.path.exists(src): shutil.move(src, dest_f)

            # 空フォルダ掃除
            for root, dirs, files in os.walk(raw_dir, topdown=False):
                for name in dirs:
                    try:
                        d_path = os.path.join(root, name)
                        if not os.listdir(d_path): os.rmdir(d_path)
                    except: pass

            with open(esd_path, 'w', encoding='utf-8') as f:
                f.writelines(new_esd_lines)

            self.log(f"完了しました。\n削除: {deleted_count}, 移動: {moved_count}")
            messagebox.showinfo("完了", "整理が完了しました。")
            self.load_preview()

        except Exception as e:
            messagebox.showerror("エラー", f"処理中にエラーが発生しました: {e}")
            self.log(f"エラー: {e}")

    # ---------------------------------------------------------
    # 重複音声検出機能 (Deduplication Tool)
    # ---------------------------------------------------------
    def start_dedup_check(self):
        dataset_root = self.dataset_dir.get()
        if not dataset_root or not os.path.exists(dataset_root):
            messagebox.showerror("エラー", "データセットフォルダを指定してください。")
            return
        
        esd_path = os.path.join(dataset_root, "esd.list")
        raw_dir = os.path.join(dataset_root, "raw")
        
        if not os.path.exists(esd_path) or not os.path.exists(raw_dir):
            messagebox.showerror("エラー", "esd.list または rawフォルダが見つかりません。")
            return
        
        # パラメータ取得
        try:
            p_top_db = self.dedup_top_db.get()
            p_hash_sr = self.dedup_hash_sr.get()
            p_q_levels = self.dedup_q_levels.get()
        except:
            messagebox.showerror("エラー", "パラメータには数値を入力してください。")
            return

        msg = (
            "音声ファイルの内容を解析して重複チェックを行います。\n\n"
            f"設定:\n Top DB: {p_top_db}\n Hash SR: {p_hash_sr}\n Q Levels: {p_q_levels}\n\n"
            "ファイル数によっては数分以上かかる場合があります。\n開始しますか？"
        )

        # スレッドで処理開始
        if messagebox.askyesno("確認", msg):
            self.log(f"=== 重複チェック開始 (SR={p_hash_sr}, Q={p_q_levels}, DB={p_top_db}) ===")
            threading.Thread(
                target=self.run_dedup_analysis, 
                args=(esd_path, raw_dir, p_top_db, p_hash_sr, p_q_levels), 
                daemon=True
            ).start()

    def run_dedup_analysis(self, esd_path, raw_dir, top_db, hash_sr, q_levels):
        """重複検出ロジック（dedup_esd.py ベース）"""
        try:
            records = []
            with open(esd_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line: continue
                    parts = line.split("|", 3)
                    if len(parts) != 4: continue
                    # (raw_line, filename, actor, lang, text)
                    records.append((line, parts[0], parts[1], parts[2], parts[3]))

            # テキストごとにグループ化
            text_groups = defaultdict(list)
            for rec in records:
                text = rec[4]
                text_groups[text].append(rec)

            self.log(f"テキストグループ数: {len(text_groups)}")
            
            duplicate_groups = [] # [ {text: str, keep: rec, duplicates: [rec, ...]} ]
            total_checked = 0
            
            for text, group in text_groups.items():
                if len(group) < 2:
                    total_checked += 1
                    continue
                
                # 同じテキスト内で内容ハッシュ比較
                hash_to_first = {} # hash -> record
                current_group_dupes = [] # (original_rec, duplicate_rec)

                for rec in group:
                    # parts[0] might be full path or relative
                    filename = os.path.basename(rec[1])
                    # rawフォルダ以下を探索 (サブフォルダ対応)
                    # 高速化のため、すでにraw_files_mapがあればそれを使う、なければ探す
                    audio_path = self.find_audio_path(raw_dir, filename)

                    if not audio_path:
                        self.log(f"[WARN] File not found: {filename}")
                        continue
                    
                    # ハッシュ計算 (パラメータ適用)
                    h = audio_content_hash(
                        audio_path, 
                        top_db=top_db, 
                        hash_sr=hash_sr, 
                        q_levels=q_levels
                    )

                    if h is None:
                        continue
                    
                    if h in hash_to_first:
                        # 重複発見
                        original = hash_to_first[h]
                        current_group_dupes.append((original, rec, audio_path)) # (KEEP, DELETE, DELETE_PATH)
                    else:
                        hash_to_first[h] = rec
                        # Keep側も再生したいのでパスを保存しておく
                        # original変数はレコードタプルなので、ここでハッシュマップの値を拡張してパスも持たせるのが良いが、
                        # 簡易的に、hash_to_first[h] = (rec, audio_path) に変更する
                        hash_to_first[h] = (rec, audio_path)
                    
                    total_checked += 1
                    if total_checked % 50 == 0:
                        self.log_area.after(0, self.update_log_msg, f"解析中... {total_checked}/{len(records)}")

                if current_group_dupes:
                    # このテキストグループでの重複情報をまとめる
                    # 重複ごとにエントリを追加
                    for orig_data, dupe, d_path in current_group_dupes:
                        orig_rec = orig_data[0]
                        orig_path = orig_data[1]
                        
                        duplicate_groups.append({
                            "text": text,
                            "keep_file": os.path.basename(orig_rec[1]),
                            "keep_path": orig_path,
                            "delete_file": os.path.basename(dupe[1]),
                            "delete_path": d_path,
                            "delete_line": dupe[0],
                            "keep_line": orig_rec[0],
                            "selected": True  # 初期値ON
                        })

            self.log(f"解析完了。重複候補: {len(duplicate_groups)}件")
            
            # 結果表示をメインスレッドで行う
            self.root.after(0, lambda: self.show_dedup_results(duplicate_groups, esd_path, raw_dir))

        except Exception as e:
            self.log(f"重複チェックエラー: {e}")
            import traceback
            traceback.print_exc()

    def update_log_msg(self, msg):
        # スレッドからログ更新するためのヘルパー
        # 既存のログの最後を書き換える簡易実装
        pass 

    def find_audio_path(self, raw_dir, filename):
        # すでにマップがあればそれを使う
        if filename in self.raw_files_map:
            return self.raw_files_map[filename]
        # なければ探索
        for root, dirs, files in os.walk(raw_dir):
            if filename in files:
                path = os.path.join(root, filename)
                self.raw_files_map[filename] = path
                return path
        return None

    def show_dedup_results(self, duplicates, esd_path, raw_dir):
        """重複結果を表示するウィンドウ（チェックボックス＆再生機能付き）"""
        if not duplicates:
            messagebox.showinfo("結果", "重複音声は見つかりませんでした。\n(同一テキストで、かつ音声波形の内容が一致するもの)")
            return

        win = tk.Toplevel(self.root)
        win.title(f"重複検出結果: {len(duplicates)}件")
        win.geometry("1000x600")

        lbl = tk.Label(win, text="検出された重複ファイルです。左のチェックが入っているものが削除されます。\nファイル名をクリックすると音声を再生できます。", pady=10)
        lbl.pack()

        # Treeview
        # columns = ("Select", "Text", "KEEP", "DELETE")
        cols = ("Select", "Text", "KEEP (Original)", "DELETE (Duplicate)")
        tree = ttk.Treeview(win, columns=cols, show='headings')
        
        tree.heading("Select", text="削除")
        tree.column("Select", width=60, anchor="center")
        
        tree.heading("Text", text="Text")
        tree.column("Text", width=300)
        
        tree.heading("KEEP (Original)", text="KEEP (Original) [Click to Play]")
        tree.column("KEEP (Original)", width=280)
        
        tree.heading("DELETE (Duplicate)", text="DELETE (Duplicate) [Click to Play]")
        tree.column("DELETE (Duplicate)", width=280)

        vsb = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(fill="both", expand=True, padx=10)
        vsb.pack(side="right", fill="y")

        # データを保持して紐付けるためのID
        self.dedup_data_map = {} 

        # 文字列によるチェックボックス表現
        CHECKED = "☑"
        UNCHECKED = "☐"

        for i, d in enumerate(duplicates):
            d["selected"] = True # 初期値
            iid = tree.insert("", "end", values=(CHECKED, d["text"], d["keep_file"], d["delete_file"]))
            self.dedup_data_map[iid] = d

        # クリックイベントハンドラ
        def on_dedup_tree_click(event):
            region = tree.identify_region(event.x, event.y)
            if region != "cell": return
            
            column = tree.identify_column(event.x)
            row_id = tree.identify_row(event.y)
            
            if not row_id: return
            
            data = self.dedup_data_map.get(row_id)
            if not data: return

            # カラム判定 (#1=Select, #2=Text, #3=Keep, #4=Delete)
            if column == "#1":
                # チェック切り替え
                current_val = data["selected"]
                new_val = not current_val
                data["selected"] = new_val
                
                # 表示更新
                new_icon = CHECKED if new_val else UNCHECKED
                # valuesを更新
                current_values = list(tree.item(row_id, "values"))
                current_values[0] = new_icon
                tree.item(row_id, values=current_values)
            
            elif column == "#3":
                # KEEP 再生
                if os.path.exists(data["keep_path"]):
                    self.play_audio(data["keep_path"])
                else:
                    print(f"File not found: {data['keep_path']}")

            elif column == "#4":
                # DELETE 再生
                if os.path.exists(data["delete_path"]):
                    self.play_audio(data["delete_path"])
                else:
                    print(f"File not found: {data['delete_path']}")

        tree.bind("<Button-1>", on_dedup_tree_click)

        # アクションボタン
        frame_btn = tk.Frame(win, pady=10)
        frame_btn.pack(fill="x")

        def do_delete():
            # 選択されているものだけ抽出
            selected_dupes = [d for d in duplicates if d["selected"]]
            
            count = len(selected_dupes)
            if count == 0:
                messagebox.showinfo("情報", "削除対象が選択されていません。")
                return

            if not messagebox.askyesno("警告", f"チェックされた {count} 件のファイルを削除しますか？\n・バックアップは作成されます\n・重複ファイルは物理削除されます"):
                return
            
            win.destroy()
            self.execute_dedup_deletion(selected_dupes, esd_path, raw_dir)

        btn_del = tk.Button(frame_btn, text="選択した項目を削除してesd.listを更新", command=do_delete, bg="#ffcdd2", font=("bold", 10), height=2)
        btn_del.pack(side="right", padx=20)
        
        btn_cancel = tk.Button(frame_btn, text="キャンセル", command=win.destroy)
        btn_cancel.pack(side="right")

    def execute_dedup_deletion(self, duplicates, esd_path, raw_dir):
        """重複削除の実行"""
        try:
            self.log("=== 重複削除処理開始 ===")
            
            # バックアップ
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_root = os.path.dirname(esd_path) + f"/backup_dedup_{timestamp}"
            os.makedirs(backup_root, exist_ok=True)
            shutil.copy2(esd_path, os.path.join(backup_root, "esd.list"))
            # 重複削除の場合はRaw全バックアップは重すぎる可能性もあるが、安全のためやる
            # しかし容量不足のリスクもあるため、今回はesdバックアップのみにし、ファイル削除は慎重に行う
            # または「削除されるファイルのみ」バックアップするか？
            # ここでは安全重視で「esdのみバックアップ」+「削除ファイルはゴミ箱へ(実装困難)」
            # -> 簡易的に、削除対象ファイルをバックアップフォルダに退避(move)する形にする
            backup_files_dir = os.path.join(backup_root, "deleted_files")
            os.makedirs(backup_files_dir, exist_ok=True)
            
            self.log(f"バックアップ作成: {backup_root}")

            # 削除対象の行セットを作成
            lines_to_remove = set(d["delete_line"] for d in duplicates)
            files_to_remove = set(d["delete_path"] for d in duplicates)

            # esd.list 再構築
            new_lines = []
            with open(esd_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip() in lines_to_remove:
                        continue # 除外
                    new_lines.append(line)
            
            # ファイル書き込み
            with open(esd_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            
            # 音声ファイル削除 (退避)
            deleted_count = 0
            for f_path in files_to_remove:
                if os.path.exists(f_path):
                    fname = os.path.basename(f_path)
                    shutil.move(f_path, os.path.join(backup_files_dir, fname))
                    deleted_count += 1
            
            self.log(f"処理完了。\n更新されたesd行数: {len(new_lines)}\n削除(退避)されたファイル: {deleted_count}件")
            messagebox.showinfo("完了", f"重複削除が完了しました。\n削除されたファイルは以下に退避されています:\n{backup_files_dir}")

        except Exception as e:
            messagebox.showerror("エラー", f"削除処理中にエラーが発生しました: {e}")
            self.log(f"エラー: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetOrganizerApp(root)
    root.mainloop()