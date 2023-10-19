import numpy as np
import faiss
import glob
import json
import matplotlib.pyplot as plt
import os, sys
import math
import clip
import torch
import time
import pandas as pd
import re
from translate_processing import Translation
from langdetect import detect
from pathlib import Path

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

mode_compute = 'clip'
keyframes_id_path = '/home/hoangtv/Desktop/Attention/txt2vid_ret/data/dict/keyframes_id_path.json'
bin_path = '/home/hoangtv/Desktop/Attention/txt2vid_ret/data/bins/clip_norm_faiss_cosine.bin'
# for results
result_path = os.path.join(WORK_DIR, 'results')
mode_result_path = os.path.join(result_path, mode_compute)


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


class CLIPSearch:

    def __init__(self, features_path=bin_path, keyframes_dict=keyframes_id_path, mode='clip', show_time_compute=True):
        self.mode = mode
        self.show_time_compute = show_time_compute
        self.index = faiss.read_index(features_path)
        self.translator = Translation()
        self.keyframes_id = self.load_dict_from_json_file(keyframes_dict) # read keyframes_id.json
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model, preprocess = clip.load("ViT-B/16", device=self.device)


    def text_search(self, text, k):
        ###### TEXT FEATURES EXACTING ######
        text = clip.tokenize([text]).to(self.device)  
        text_features = self.model.encode_text(text).cpu().detach().numpy().astype(np.float32)

        ###### SEARCHING #####
        scores, idx_image = self.index.search(text_features, k=k)
        idx_image = idx_image.flatten()
        image_paths = [self.keyframes_id[f"{str(i)}"] for i in idx_image]

   
        return scores, idx_image, image_paths
    


    def load_dict_from_json_file(self, json_path: str):
        with open(json_path, 'r') as f:
            js = json.loads(f.read())
        return js