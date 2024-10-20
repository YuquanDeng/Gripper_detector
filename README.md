# Gripper_detector

## installation

1. create an environment
```
conda create --name gripper_det python=3.8.1
conda activate gripper_det
git clone this repo
```
2. install torch
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
3. you maybe need to degrade gcc bulabula:

```
sudo apt install gcc-9 g++-9 gcc-10 g++-10 gcc-11 g++-11 g++-12 gcc-12 g++-13 gcc-13 g++-14 gcc-14 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 70 --slave /usr/bin/g++ g++ /usr/bin/g++-11 --slave /usr/bin/gcov gcov /usr/bin/gcov-11
```

4. install off-the-shelf detectron2
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```


## inference
just run:
``` python ./batch_inference.py```

see parser in ```batch_inference.py``` to control if visualize and text_promt and image folder

## supported objects 
We now only support:
```
self.supported_objs = {0: {'id': 0, 'name': 'blue cube', 'color': [220, 20, 60], 'isthing': 1},
                               1: {'id': 1, 'name': 'green cube', 'color': [220, 20, 60], 'isthing': 1},
                               2: {'id': 2, 'name': 'magenta cube', 'color': [220, 20, 60], 'isthing': 1},
                               3: {'id': 3, 'name': 'orange cube', 'color': [220, 20, 60], 'isthing': 1},
                               4: {'id': 4, 'name': 'purple cube', 'color': [220, 20, 60], 'isthing': 1},
                               5: {'id': 5, 'name': 'red cube', 'color': [220, 20, 60], 'isthing': 1},
                               6: {'id': 6, 'name': 'yellow cube', 'color': [220, 20, 60], 'isthing': 1}}
```
which means your ```text_prompt``` must be one of them, we can further train a more diverse selector, but the current one should be enough to experiment on "pick cube".

## some demos
I have run some demos, the results are in ```/scratch/partial_datasets/rlbench/bimanual_data/det_att_pred```