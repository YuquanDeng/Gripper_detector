import json
import numpy as np

# Path to your JSON file
# json_path = '/data/input/jiafei/GroundedVLA/data/libero/libero_10_no_noops_line.json' # TODO
json_path = "/home/nil/manipulation/dataset/pick_plum_100/pt_json/rgb.json"

# Load JSON data
with open(json_path, 'r') as f:
    data = json.load(f)

# Group samples by episode.
# Assuming the image path format is like:
#   /gscratch/krishna/jason328/openvla/collection/02-26-25/001/003.png
# where the folder before the filename is the episode id.
episodes = {}
for sample in data:
    # Split the path and extract the episode folder (penultimate element)
    parts = sample['image'].split('/')
    episode = parts[-1][:parts[-1].find("step_")-1]
    # Also extract the step number from the filename (e.g., "003" from "003.png")
    # step = int(parts[-1].split('.')[0]) # TODO
    # sample['episode'] = episode # TODO
    # sample['step'] = step
    step = int(parts[-1][parts[-1].find("step_"):].replace(".png", "").replace("step_", ""))
    sample['episode'] = episode
    sample['step'] = step
    episodes.setdefault(episode, []).append(sample)

# Sort each episode's samples by step number
for episode in episodes:
    episodes[episode].sort(key=lambda x: x['step'])

# Prepare output list
output = []

# Process each sample
for episode, samples in episodes.items():
    for i, sample in enumerate(samples):
        # Get the trajectory from the current sample to the last sample of the episode
        traj = samples[i:]
        n = len(traj)

        if n < 5:
            # Use all available points
            line = [s['point'] for s in traj]
            # Pad with the last point until the line has exactly 5 points
            while len(line) < 5:
                line.append(line[-1])
        else:
            # Otherwise, sample 5 points evenly along the trajectory:
            # starting at the current point and ending at the final point
            indices = np.linspace(0, n - 1, num=5, dtype=int)
            line = [traj[idx]['point'] for idx in indices]

        # Append the result for the current sample
        output.append({
            'image': sample['image'],
            'line': line
        })

save_json_path = json_path[:-5] + "_line.json"

# Optionally, save the output to a new JSON file
with open(save_json_path, 'w') as f:
    json.dump(output, f, indent=4)

# For debugging, print the output
for item in output:
    print(item)