import os
import argparse
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_scalars(logdir, output_file='tb_results.csv'):
    # スカラーデータを格納するリスト
    all_scalar_data = []

    # ログディレクトリ内の全ファイルを探索
    for root, dirs, files in os.walk(logdir):
        for file in files:
            if 'tfevents' in file:
                path = os.path.join(root, file)
                print(f"Processing: {path}")
                
                try:
                    ea = EventAccumulator(path)
                    ea.Reload()
                    
                    # 保存されているすべてのスカラータグを取得
                    tags = ea.Tags()['scalars']
                    
                    for tag in tags:
                        events = ea.Scalars(tag)
                        for event in events:
                            all_scalar_data.append({
                                'run_name': os.path.relpath(root, logdir), # フォルダ構成をRun名とする
                                'tag': tag,
                                'step': event.step,
                                'value': event.value,
                                'wall_time': event.wall_time
                            })
                except Exception as e:
                    print(f"Error processing {path}: {e}")

    # DataFrameに変換
    if not all_scalar_data:
        print("スカラーデータが見つかりませんでした。")
        return

    df = pd.DataFrame(all_scalar_data)
    
    # AIが読みやすい形式（ピボットテーブル）に整形
    # 行：Step, 列：Run名+タグ名
    df['run_tag'] = df['run_name'] + "/" + df['tag']
    pivot_df = df.pivot_table(index='step', columns='run_tag', values='value')
    
    # CSV保存
    pivot_df.to_csv(output_file)
    print(f"Done! Saved to {output_file}")
    print("このCSVファイルの中身、または先頭・末尾のデータをAIに共有してください。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True, help="Path to tensorboard log directory")
    args = parser.parse_args()
    
    extract_scalars(args.logdir)