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


    def text_search(self, text, k):
        text = InternVideo.tokenize(text).cuda()
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features = torch.nn.functional.normalize(text_features, dim=1).cpu().detach().numpy().astype(np.float32)
        scores, idx_scene = self.index.search(text_features, k=k)
        # print(f"idx scene shape from text search: {idx_scene.shape}")

        idx_scene = idx_scene.flatten()

        # No need to return image paths, the paths will be processed later
        # image_paths = [self.keyframes_id[str(i)] for i in idx_image]
        return scores, idx_scene

    
    def load_dict_from_json_file(self, json_path: str):
        with open(json_path, 'r') as f:
            js = json.loads(f.read())
        return js


    def load_index_from_bin_file(self, bin_file=bin_path):
        self.index = faiss.read_index(bin_file)


    # @time_complexity
    def show_images_irv(self, scene_indexes, query, timestamp, save_path, method='text'):
        counter = 0 # biến đếm, chỉ dùng để xác định vị trí add subplot
        # Sẽ được thay thế khi có phương pháp plot nhiều subplots trong 1 lần 

        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(f"Query: {query}")
        # columns = int(math.sqrt(len(images)))
        # rows = int(np.ceil(len(images)/columns))
        columns = 5
        rows = 8 # k=10 và hiển thị 4 => 40 = rows*column, chưa generalize

        for index in scene_indexes:
            # Prepare data from query results for loading
            video_name = self.scene_dict[str(index)]["video"]
            video_path = self.scene_dict[str(index)]["video_path"]
            start = self.scene_dict[str(index)]["start"]
            end = self.scene_dict[str(index)]["end"]

            # Load results
            imgs, ids = InternVideo.load_results(video_path, start, end)

            # Plot loaded results
            for img, id in zip(imgs, ids):
                img = img
                ax = fig.add_subplot(rows, columns, counter+1)
                ax.set_title(f"{video_name}, {id:0>6d}")
                counter += 1

                plt.imshow(img)
                plt.axis("off")
            # plt.title(f"Query: {query}")
        plt.savefig(os.path.join(save_path, f'{timestamp}_{method}_retrieval.jpg'))
        plt.show()


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
        

    def scene_to_img_mapping(self, scene_idxs):
        img_paths = []
        for idx in scene_idxs:
            img_paths.extend(self.scene_dict[str(idx)]["keyframes"])
        return img_paths
    
    def submit(self, img_paths, file_name_save='test',select_ids = ""):
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

    irv_search = IRVSearch(bin_file=bin_path, scenes_dict_path=scenes_dict_path)
    print('>>> Init done. Info:')
    print(f'\tNumber of features from bin: {irv_search.index.ntotal}')
    
    while True:
        input_text = input("Enter your query: ")
        translated_text = irv_search.translator(input_text)
        print(f"Translated text: {translated_text}")
        scores, scene_idx = irv_search.text_search(translated_text, k=25)

        timer = time.time()
        img_paths = irv_search.scene_to_img_mapping(scene_idx)

        print(f"Number of searched images: {len(img_paths)}")
        irv_search.show_images(img_paths, query=input_text, timestamp=time.time(), save_path=mode_result_path)
        
        select_id = input("Enter ID image to write file: ")
        csv_filename = input("Enter name of csv file: ")
        
        irv_search.submit(img_paths, csv_filename,select_id)
        print(f"Time for plotting (image-based): {time.time() - timer}")

        print('======== End of Program =========')



if __name__ == "__main__":
    main()