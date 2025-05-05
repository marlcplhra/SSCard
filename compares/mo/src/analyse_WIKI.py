import os
from datetime import timedelta

def time_to_seconds(time_str):
    time_parts = time_str.split(":")
    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    seconds = float(time_parts[2])
    return hours * 3600 + minutes * 60 + seconds

def process_files_in_folder(folder_path, K):
    total_building_time = 0.0
    total_model_size = 0.0
    
    for i in range(K):
        file_path = os.path.join(folder_path, f"{i}_top1/building_results.txt")
    # for folder in os.listdir(folder_path):
    #     file_path = os.path.join(folder_path, folder, 'building_results.txt')
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
                for line in lines:
                    if "Building time" in line:
                        building_time_str = line.split(":", 1)[1].strip()
                        total_building_time += time_to_seconds(building_time_str)
                    elif "Model size" in line:
                        model_size = float(line.split(":")[1].strip())
                        total_model_size += model_size

    return total_building_time, total_model_size

folder_path = "../exp/WIKI"


total_building_time, total_model_size = process_files_in_folder(folder_path, K=10)


print(f"Total Building Time: {str(timedelta(seconds=total_building_time))}")
print(f"Total Model Size: {total_model_size} MB")
