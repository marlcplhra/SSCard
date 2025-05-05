import os
import shutil
import pandas as pd


def create_folder():
    TARGET_DIR = "./"

    # 遍历目标文件夹中的所有文件
    for file in os.listdir(TARGET_DIR):
        file_path = os.path.join(TARGET_DIR, file)
        
        # 只处理txt文件
        if os.path.isfile(file_path) and file.endswith('.txt'):
            fname, _ = os.path.splitext(file)
            new_folder_path = os.path.join(TARGET_DIR, fname)
            
            # 创建新的文件夹（如果不存在）
            os.makedirs(new_folder_path, exist_ok=True)
            
            # 移动文件到新建的文件夹里
            new_file_path = os.path.join(new_folder_path, file)
            shutil.move(file_path, new_file_path)

    print("处理完成！")



def sample_and_save_csvs(root_folder, output_folder, max_rows=500_000):
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.csv'):
                if 'sampled' in file: continue
                file_path = os.path.join(subdir, file)
                print(f"Processing: {file_path}")

                # Read CSV
                # df = pd.read_csv(file_path)
                df = pd.read_csv(file_path, na_values=[], keep_default_na=False)

                # Sample if necessary
                if len(df) > max_rows:
                    df_sampled = df.sample(n=max_rows, random_state=42)
                else:
                    df_sampled = df

                # Generate relative path to maintain folder structure
                relative_path = os.path.relpath(subdir, root_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                # if not os.path.exists(output_subfolder):
                #     os.makedirs(output_subfolder)

                filename = file.rsplit('.', 1)[0]
                output_file_path = os.path.join(output_subfolder, filename + "_sampled.csv")
                if 'counts' in filename:
                    df = df.sort_values(by="string")
                    df_sampled.to_csv(output_file_path, index=False, header=True)
                else:
                    df_sampled.to_csv(output_file_path, index=False, header=True)
                    
                print(f"Saved to: {output_file_path}\n")

if __name__ == "__main__":
    # 指定你的输入文件夹和输出文件夹
    input_folder = "./"
    output_folder = "./"

    sample_and_save_csvs(input_folder, output_folder)
