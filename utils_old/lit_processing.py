from translate_processing import Translation
from langdetect import detect
from PIL import Image
from vit_jax import models
from tqdm import tqdm
from pathlib import Path
import torch
import faiss
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import math
import sys, os
import time

FILE = Path(__file__).resolve()
# Read folder containing file path
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
# main work directory
WORK_DIR = os.path.dirname(ROOT)

res = faiss.StandardGpuResources()  
VECTOR_DIMS = 768

mode_compute = 'lit'
# for keyframes dictionary
data_path = os.path.join(WORK_DIR, 'data')
keyframes_id_path = os.path.join(data_path, 'dict/keyframes_id_path.json')
# for features file
folder_features = os.path.join(data_path, 'bins')
bin_path = os.path.join(folder_features, f'{mode_compute}_faiss_cosine.bin')
# for results
result_path = os.path.join(WORK_DIR, 'results')
mode_result_path = os.path.join(result_path, mode_compute)


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


class LitSearch(Translation):
    
    def __init__(self, folder_features=folder_features, keyframes_dict=keyframes_id_path, mode='lit', show_time_compute=True):
        super(LitSearch, self).__init__()
        self.mode = mode
        self.show_time_compute = show_time_compute
        self.index = None
        # self.gpu_index = None
        self.folder_features = folder_features 
        self.keyframes_id = self.load_dict_from_json_file(keyframes_dict) # read keyframes_id.json
        # self.query = {'encoder': [], 'k': 1}   
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        
        model_name = ['LiT-B16B', 'LiT-L16L', 'LiT-L16S', 'LiT-L16Ti']
        self.lit_model = models.get_model(model_name[0])
        self.tokenizer = self.lit_model.get_tokenizer()
        self.lit_variables = self.lit_model.load_variables()
        self.image_preprocessing = self.lit_model.get_image_preprocessing()


    @time_complexity
    def text_search(self, text, k):
        text_features = self.get_mode_extract(text, method='text')
        text_features = np.array(text_features)  # Reshape to 2D array
        # gpu_index = faiss.index_cpu_to_all_gpus(self.cpu_index)
        scores, idx_image = self.index.search(text_features, k=k)
        print(f"From text search: {idx_image.shape}")

        # No flatten here because it's not necessary
        # idx_image = idx_image.flatten()

        # No need to return image paths, the paths will be processed later
        # image_paths = [self.keyframes_id[str(i)] for i in idx_image]

        return scores, idx_image


    def embed_images(self, images):
        zimg, _, _ = self.lit_model.apply(self.lit_variables, images=images)
        return zimg

    @time_complexity
    def image_search_by_id(self, image_id, k):
        query_feats = self.cpu_index.reconstruct(image_id).reshape(1,-1)
        # gpu_index = faiss.index_cpu_to_all_gpus(self.cpu_index)
        scores, idx_image = self.cpu_index.search(query_feats, k=k)
        idx_image = idx_image.flatten()

        # image_paths = [self.keyframes_id[str(item)]['image_path'] for item in list(idx_image)]
        # image_paths = list(map(lambda x: x.replace("Database", 'data/news'), image_paths))
        image_paths = [self.keyframes_id[i] for i in idx_image]
        
        return scores, idx_image, image_paths
    

    @time_complexity
    def image_search_by_path(self, image_path, k):
        image_features = self.get_mode_extract(image_path)
        image_features = np.array(image_features)
        scores, idx_image = self.cpu_index.search(image_features, k=k)

        idx_image = idx_image.flatten()

        # image_paths = [self.keyframes_id[str(item)]['image_path'] for item in list(idx_image)]
        # image_paths = list(map(lambda x: x.replace("Database", 'data/news'), image_paths))
        image_paths = [self.keyframes_id[i] for i in idx_image]
        return scores, idx_image, image_paths

    def get_mode_extract(self, data, method='text'):
        if os.path.isfile(data) and method != 'image':
            print('Set mode to image because of the image path from user !')
            method = 'image'
        
        if method == 'text':
            if detect(data) == 'vi':
                text = Translation.__call__(self, data)
            else:
                text = data # if not Vietnamese then remain the same
            if self.mode == 'lit':
                print('Translated text: ', text)
                tokens = self.tokenizer([text])
                _, data_embedding, _ = self.lit_model.apply(self.lit_variables, tokens=tokens)
            else:
                print(f'Not found model {self.mode}')


        elif method == 'image':
            if self.mode == 'lit':
                print('Choosing mode LiT')
                image = self.image_preprocessing([Image.open(data)])
                data_embedding = self.embed_images(image)
            elif self.mode == 'clip':
                image = self.preprocess(Image.open(data)).unsqueeze(0).to(self.device)
                data_embedding = self.model_clip.encode_text(data)
                data_embedding = data_embedding.cpu().detach().numpy.astype(np.float32)
            else:
                print(f'Not found model {self.mode}')

        else:
                print(f'Not found method {method}')
                
        return data_embedding
    
    
    def load_dict_from_json_file(self, json_path: str):
        with open(json_path, 'r') as f:
            js = json.loads(f.read())
        return js


    def load_index_from_bin_file(self, bin_file=bin_path):
        self.index = faiss.read_index(bin_file)


    # @time_complexity
    def show_images(self, image_paths, save_path, timestamp, method='text'):
        fig = plt.figure(figsize=(15, 15))
        # columns = int(math.sqrt(len(image_paths)))
        columns = image_paths.shape[1]
        rows = image_paths.shape[0]
        image_paths = image_paths.flatten()
        # columns = 4
        # rows = 10
        # rows = int(np.ceil(len(image_paths)/columns))
        

        for i in range(1, columns*rows +1):
            img = plt.imread(image_paths[i - 1])
            ax = fig.add_subplot(rows, columns, i)
            ax.set_title('/'.join(image_paths[i - 1].split('/')[-2:]).replace('/',',').replace('.jpg',f' {i}'))

            plt.imshow(img)
            plt.axis("off")
            plt.savefig(os.path.join(save_path, f'{timestamp}_{method}_retrieval.jpg'))
        plt.show()


    def get_subsequent_paths(self, ids):
        """
        Get ids and return an array of (ids, ids+1, id+2 , id+3)
        """
        n_columns = 4
        sub_ids = np.zeros(((ids.shape)[1], n_columns), dtype=np.uint32)
        sub_paths = np.zeros(((ids.shape)[1], n_columns), dtype=object)
        for column in range(n_columns):
            sub_ids[:, column] = ids[0] + 2*column 
            for row in range(sub_paths.shape[0]):
                sub_paths[row, column] = self.keyframes_id[str(sub_ids[row, column])]
        return sub_paths


def main():

    if not os.path.exists(os.path.join(result_path)):
        os.makedirs(result_path)
    
    if not os.path.exists(os.path.join(mode_result_path)):
        os.makedirs(mode_result_path)

    lit_search = LitSearch()
    lit_search.load_index_from_bin_file()
    print(f"Load index done, number of features: {lit_search.index.ntotal}")
    
    while True:
        text = input("Enter your query: ")
        scores, images_id = lit_search.text_search(text, k=10)
        # print(f"score: {scores.shape}")
        # print(f"image_id: {images_id}")
        # print(f"paths: {image_paths}")
        sub_paths = lit_search.get_subsequent_paths(ids=images_id)

        # print(f"image_paths: {type(image_paths)}")
        df_text = pd.DataFrame({'images_id': list(images_id), 'scores': scores[0]})
        csv_filename = input("Enter name of query output: ")
        df_text.to_csv(os.path.join(mode_result_path, csv_filename))
        # lit_search.show_images(sub_paths, mode_result_path, timestamp=time.time(), method='text')
        print('======== End of Program =========')

if __name__ == "__main__":
    main()