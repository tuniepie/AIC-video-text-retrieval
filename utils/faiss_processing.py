from tqdm import tqdm
from translate_processing import Translation
import InternVideo
import torch
import faiss
import faiss.contrib.torch_utils
import numpy as np
import json
import glob
from tqdm import tqdm


res = faiss.StandardGpuResources()  
VECTOR_DIMS = 768
model_path = '/content/drive/MyDrive/src/AIC2023/txt2vid_retrieval/models/InternVideo-MM-L-14.ckpt'
video_data_root ='/content/drive/MyDrive/datasets/AIC2023/Video'
features_bin_file = '/content/drive/MyDrive/src/AIC2023/txt2vid_retrieval/data/bins/internscene_cos_faiss.bin'
scene_txt_root = '/content/drive/MyDrive/data1/AIC2023/txt_files'
dest_bin_path ='/content/drive/MyDrive/src/AIC2023/txt2vid_retrieval/data/bins/internscene_cos_faiss.bin'
dest_dict_path = '/content/drive/MyDrive/src/AIC2023/txt2vid_retrieval/data/dicts/scenes_dict.json'

class AttentionFaiss(Translation):
    # def __init__(self, video_data_root, scene_txt_root, features_bin_file, scenes_dict_file):
    def __init__(self, video_data_root, scene_txt_root, features_bin_file):

        super(AttentionFaiss, self).__init__()
        self.video_data_root = video_data_root
        self.scene_txt_root = scene_txt_root
        # self.scenes_dict = self.load_json_file(scenes_dict_file)
        self.scene_dict_infos = []
        self.model = InternVideo.load_model(model_path).cuda()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.index = faiss.GpuIndexFlatL2(res, VECTOR_DIMS)
            self.index_gpu_flag = True
            print("Index initialized on GPU")
        else:
            self.device = torch.device("cpu")
            self.index = faiss.IndexFlatL2(VECTOR_DIMS)
            self.index_gpu_flag = False
            print("Index initialized on CPU")

    def indexing(self, des_bin_path, des_scn_dict_path):
        video_paths = sorted(glob.glob(f'{self.video_data_root}/**/*.mp4', recursive=True))
        print(video_paths)
        scene_txt_paths = sorted(glob.glob(f'{self.scene_txt_root}/**/*.txt', recursive=True))
        print(scene_txt_paths)
        for video_path, scene_txt_path in tqdm(zip(video_paths, scene_txt_paths), desc='indexing...'):
            scenes, infos = InternVideo.load_scenes(video_path, scene_txt_path)
            self.scene_dict_infos.extend(infos)
            for i, scene in enumerate(scenes):
                scene = scene.cuda()
                video_features = self.model.encode_video(scene.unsqueeze(0))
                video_features = torch.nn.functional.normalize(video_features, dim=1)
                print(f"scene {i} encoded")
                self.index.add(video_features)
        if self.index_gpu_flag:
            self.index = faiss.index_gpu_to_cpu(self.index)
        faiss.write_index(self.index, des_bin_path)
        print(f'Saving indices finish! Destination: {des_bin_path}')

        scenes_dict = dict(enumerate(self.scene_dict_infos))
        with open(des_scn_dict_path, "w") as f:
            f.write(json.dumps(scenes_dict))
            print(f"Saving dict finish! Destination: {des_scn_dict_path}")

    def load_json_file(self, json_path: str):
        with open(json_path, 'r') as f:
            js = json.loads(f.read())
        return js

    def load_index_from_bin_file(self, bin_file):
        self.index = faiss.load_index(bin_file)

    # def write_scenes_dict(self, video_name, start, end):
        # infos = []
        # video_name = txt_path.split('/')[-1].replace('.txt', '')
        # for start, end in list_scenes:
        # info = {
        #     "video": video_name,
        #     "start": f'{start:0>6d}',
        #     "end": f'{end:0>6d}'
        # }
        # self.infos.append(info)
        # scenes_dict = dict(enumerate(self.infos))
        # with open('/content/drive/MyDrive/src/AIC2023/txt2vid_retrieval/\
        # data/dicts/scenes_dict.json', 'w') as f:
        #     f.write(json.dumps(scenes_dict))
        #     print('scenes dict written')

def main():
    attention_faiss = AttentionFaiss(video_data_root, scene_txt_root,features_bin_file)
    attention_faiss.indexing(dest_bin_path, dest_dict_path)

if __name__ == "__main__":
    main()