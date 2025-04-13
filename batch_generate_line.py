
import argparse
import glob 
import re
from tqdm import tqdm
import os
import subprocess

# python3 batch_generate_line.py --ckpt ckpt/model_final.pth --input_folder /home/nil/manipulation/dataset/pick_apple_100_328/rgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing RGB images")
    args = parser.parse_args()
    
    # construct cmd
    input_folders = glob.glob(args.input_folder + "/*/*/episode_*")

    def extract_episode_number(folder):
        match = re.search(r'episode_(\d+)', folder)
        return int(match.group(1)) if match else float('inf')
    
    input_folders.sort(key=extract_episode_number)
    
    for input_folder in tqdm(input_folders):
        cmd = f"python3 generate_line.py --ckpt {args.ckpt} --input_folder {input_folder}"

        log_file = input_folder.replace("rgb", "line_log")+".txt"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Execute the command and save the output to the log file
        with open(log_file, "w") as log:
            process = subprocess.run(cmd, shell=True, stdout=log, stderr=log)
            if process.returncode != 0:
                print(f"Command failed for {input_folder}. Check log file: {log_file}")
            else:
                print(f"Command succeeded for {input_folder}. Log saved to: {log_file}")
            
        # TODO: Execute the command and save the output to the log file
        print(f"Executing command: {cmd}")
        print(f"Saving output to: {log_file}")


if __name__ == "__main__":
    main()