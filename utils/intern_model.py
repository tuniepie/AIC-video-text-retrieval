import InternVideo
import numpy as np
import pandas as pd
import json
import torch
from translate_processing import Translation
import time
import faiss
from pathlib import Path
import os, sys
import matplotlib.pyplot as plt
import math
import csv

FILE = Path(__file__).resolve()
# Read folder containing file path
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
# main work directory
WORK_DIR = os.path.dirname(ROOT)

res = faiss.StandardGpuResources()  
VECTOR_DIMS = 512

mode_compute = 'irv'

# For keyframes dictionary
data_path = os.path.join(WORK_DIR, 'data')
# scenes_dict_path = os.path.join(data_path, 'dict/scenes_with_path_no_kf.json')
# Tạm thời cứ dùng những đường dẫn tuyệt đối này cho đỡ nhầm
scenes_dict_path = '/home/hoangtv/Desktop/Attention/txt2vid_ret/data/dict/scenes_with_kf_paths.json'

# For features file
folder_features = os.path.join(data_path, 'bins')
# bin_path = os.path.join(folder_features, f'{mode_compute}_faiss_cosine.bin')
# Tạm thời cứ dùng những đường dẫn tuyệt đối này cho đỡ nhầm
bin_path = '/home/hoangtv/Desktop/Attention/txt2vid_ret/data/bins/irv_faiss_cosine.bin'

# For results
result_path = os.path.join(WORK_DIR, 'results')
mode_result_path = os.path.join(result_path, mode_compute)
keyframes_root = '/media/hoangtv/New Volume/backup/data'

def time_complexity(func):
    def wrapper(*args, **kwargs):
        if args[0].show_time_compute:
            start = time.time()
            results = func(*args, **kwargs)
            print('Time requires for plotting of {}: {}'.format(args[0].mode, time.time() - start))
            return results
        else:
            return func(*args, **kwargs)
    return wrapper


class IRVSearch:

    def __init__(self, bin_file=bin_path, scenes_dict_path=scenes_dict_path, mode='irv', show_time_compute=True):
        self.mode = mode
        self.show_time_compute = show_time_compute
        self.index = faiss.read_index(bin_file)
        self.translator = Translation()
        # self.gpu_index = None
        self.scene_dict = self.load_dict_from_json_file(scenes_dict_path) # read keyframes_id.json
        # self.query = {'encoder': [], 'k': 1}   
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model = InternVideo.load_model('/home/hoangtv/Desktop/Attention/txt2vid_ret/models/InternVideo-MM-B-16.ckpt').cuda()

    def scene_to_img_mapping(self, scene_idxs):
        img_paths = []
        for idx in scene_idxs:
            img_paths.extend(self.scene_dict[str(idx)]["keyframes"])
        return img_paths

    def text_search(self, text, k):
        text = InternVideo.tokenize(text).cuda()
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features = torch.nn.functional.normalize(text_features, dim=1).cpu().detach().numpy().astype(np.float32)
        scores, idx_scene = self.index.search(text_features, k=k)


        idx_scene = idx_scene.flatten()
        image_paths = self.scene_to_img_mapping(idx_scene)


        return scores, idx_scene,image_paths

    def load_dict_from_json_file(self, json_path: str):
        with open(json_path, 'r') as f:
            js = json.loads(f.read())
        return js

    def load_index_from_bin_file(self, bin_file=bin_path):
        self.index = faiss.read_index(bin_file)


 

