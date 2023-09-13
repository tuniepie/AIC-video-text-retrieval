import InternVideo
import numpy as np
import json
import torch
from translate_processing import Translation
import time
import faiss
from pathlib import Path
import os, sys
import matplotlib.pyplot as plt
import math

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
# for keyframes dictionary
data_path = os.path.join(WORK_DIR, 'data')
# scenes_dict_path = os.path.join(data_path, 'dict/scenes_dict_full.json')
scenes_dict_path = '/home/hoangtv/Desktop/Attention/lst_scenes/scenes_dict_to_L10_final.json' # temporarily use this, wait for data batch2
# for features file
folder_features = os.path.join(data_path, 'bins')
# bin_path = os.path.join(folder_features, f'{mode_compute}_faiss_cosine.bin')
bin_path = '/home/hoangtv/Desktop/Attention/txt2vid_ret/backup/irv_faiss_cosine_L10.bin' # temporarily use this, wait for data batch2
# for results
result_path = os.path.join(WORK_DIR, 'results')
mode_result_path = os.path.join(result_path, mode_compute)
keyframes_root = '/media/hoangtv/New Volume/backup/data'

def time_complexity(func):
    def wrapper(*args, **kwargs):
        if args[0].show_time_compute:
            start = time.time()
            results = func(*args, **kwargs)
            print('Time inference of {} and base line: {}'.format(args[0].mode, time.time() - start))
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


    @time_complexity
    def text_search(self, text, k):
        text = InternVideo.tokenize(text).cuda()
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features = torch.nn.functional.normalize(text_features, dim=1).cpu().detach().numpy().astype(np.float32)
        # text_features = np.array(text_features)  # Reshape to 2D array
        # gpu_index = faiss.index_cpu_to_all_gpus(self.cpu_index)
        scores, idx_scene = self.index.search(text_features, k=k)
        print(f"idx scene shape from text search: {idx_scene.shape}")

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


    def show_images(self, image_paths, timestamp, save_path, method='text'):
        fig = plt.figure(figsize=(15, 10))
        columns = int(math.sqrt(len(image_paths)))
        rows = int(np.ceil(len(image_paths)/columns))
        
        for i in range(len(image_paths)):

            img = plt.imread(image_paths[i])
            ax = fig.add_subplot(rows, columns, i+1)
            # ax.set_title('/'.join(image_paths[i].split('/')[-2:]).replace('/',',').replace('.jpg',f' {i}'))
            ax.set_title('/'.join(image_paths[i].split('/')[-2:]).replace('/',','))
            # ax.set_title('/'.join(image_paths[i].split('/')[-2:]) + str(scores[:, i]))

            plt.imshow(img)
            plt.axis("off")
            plt.savefig(os.path.join(save_path, f'{timestamp}_{method}_retrieval.jpg'))
        plt.show()


    def scene_to_img_mapping(self, keyframes_root, scene_idxs):
        # get all keyframes ids
        img_paths = []
        for idx in scene_idxs:
            video_name = self.scene_dict[str(idx)]["video"]
            padding = 'Keyframes_' + video_name[:3]
            for frame in self.scene_dict[str(idx)]["keyframes"]:
                path = os.path.join(keyframes_root, padding, video_name, f"{frame:0>6d}.jpg")
                img_paths.append(path)

        return img_paths


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
    # print(f'\tScene dict loaded from}')
    print(f'\tNumber of features from bin: {irv_search.index.ntotal}')
    
    while True:
        input_text = input("Enter your query: ")
        translated_text = irv_search.translator(input_text)
        scores, scene_idx = irv_search.text_search(translated_text, k=10)

        img_paths = irv_search.scene_to_img_mapping(keyframes_root, scene_idx)
        irv_search.show_images(img_paths, timestamp=time.time(), save_path=mode_result_path)

        print('======== End of Program =========')



if __name__ == "__main__":
    main()