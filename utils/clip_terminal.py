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

        ###### GET INFOS KEYFRAMES_ID ######
        # infos_query = list(map(self.id2img_fps.get, list(idx_image)))
        # image_paths = [info['image_path'] for info in infos_query]
        # lst_shot = [info['list_shot_id'] for info in infos_query]

        # print(f"scores: {scores}")
        # print(f"idx: {idx_image}")
        # print(f"paths: {image_paths}")

        # return scores, idx_image, infos_query, image_paths
        return scores, idx_image, image_paths
    

    # Index reading is performed directly at initialization
    # def load_index_from_bin_file(self):
    #     bin_file = self.features_path
    #     self.index = faiss.read_index(bin_file)
        

    def load_dict_from_json_file(self, json_path: str):
        with open(json_path, 'r') as f:
            js = json.loads(f.read())
        return js
    

    def show_images(self, image_paths, query, timestamp, save_path='results', method='text'):
        fig = plt.figure(figsize=(30, 20))
        fig.suptitle(f"Query: {query}")
        columns = int(math.sqrt(len(image_paths)))
        rows = int(np.ceil(len(image_paths)/columns))
        

        for i in range(len(image_paths)):

            img = plt.imread(image_paths[i])
            ax = fig.add_subplot(rows, columns, i+1)
            ax.set_title('/'.join(image_paths[i].split('/')[-2:]).replace('/',',').replace('.jpg',f' {i}'))
            # ax.set_title('/'.join(image_paths[i].split('/')[-2:]) + str(scores[:, i]))

            plt.imshow(img)
            plt.axis("off")
        plt.savefig(os.path.join(save_path, f'{timestamp}_{method}_retrieval.jpg'))
        print(os.path.join(save_path, f'{timestamp}_{method}_retrieval.jpg'))
        plt.show()
        

    

    def submit(self, img_paths, file_name_save='',select_ids = ""):
        os.makedirs(os.path.join(WORK_DIR, 'submission'), exist_ok=True)
        video_names = [img_path.split("/")[-2] for img_path in img_paths]
        frames = [img_path.split("/")[-1].split(".")[0] for img_path in img_paths]
        if select_ids == "a":
            df_submit = pd.DataFrame({'videos': video_names,'frames': frames})
            df_submit.to_csv(os.path.join(WORK_DIR, f'submission/{file_name_save}.csv'), index=False, header=False)
        elif select_ids == "": return 0
        else:
            select_output = [int(id) for id in select_ids.strip().split(' ')]
    
            video_names = [img_paths[i].split("/")[-2] for i in select_output]
            frames = [img_paths[i].split("/")[-1].split(".")[0] for i in select_output]
            df_submit = pd.DataFrame({'videos': video_names,'frames': frames})
            df_submit.to_csv(os.path.join(WORK_DIR, f'submission/{file_name_save}.csv'), index=False, header=False)
        
       
def main():

    if os.getcwd() != WORK_DIR:
        print("Changing to proper working directory...")
        os.chdir(WORK_DIR)
        print(f"Done, working directory: {os.getcwd()}")

    if not os.path.exists(os.path.join(result_path)):
        os.makedirs(result_path)

    if not os.path.exists(os.path.join(mode_result_path)):
        os.makedirs(mode_result_path)
        
    clip_search = CLIPSearch()
    print('>>> Init done. Info:')
    print(f'\tDevice: {clip_search.device}')
    print(f'\tNumber of features from bin: {clip_search.index.ntotal}')
    
    while True:
        input_text = input("Enter your query: ")
        translated_text = clip_search.translator(input_text)
        print(f"Translated text: {translated_text}")
        scores, images_id, image_paths = clip_search.text_search(translated_text, k=100)

        timer = time.time()

        clip_search.show_images(image_paths, query=input_text, timestamp=time.time(), save_path=mode_result_path)
        
        select_id = input("Enter ID image to write file: ")
        csv_filename = input("Enter name of csv file: ")
        
        clip_search.submit(image_paths, csv_filename,select_id)
        print(f"Time for plotting (image-based): {time.time() - timer}")

        print('======== End of Program =========')


if __name__ == "__main__":
    main()