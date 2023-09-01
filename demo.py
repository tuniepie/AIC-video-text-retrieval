from utils import InternVideo

import torch
import faiss
import faiss.contrib.torch_utils
import numpy as np

res = faiss.StandardGpuResources()  
VECTOR_DIMS = 768
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")
    index = faiss.GpuIndexFlatL2(res, VECTOR_DIMS)
    index_gpu_flag = True
    print("Index initialized on GPU")
else:
    device = torch.device("cpu")
    index = faiss.IndexFlatL2(VECTOR_DIMS)
    index_gpu_flag = False
    print("Index initialized on CPU")


# bin_path =  '/content/drive/MyDrive/src/AIC2023/txt2vid_retrieval/data/dicts/test.bin'


text_cand = ["a nurse testing a patient", "an airplane is flying", "a dog is chasing a ball"]
# video = InternVideo.load_video("/content/drive/MyDrive/data1/AIC2023/video_scenes_L10/L10_V030/video_from_001488_to_001554.mp4").cuda()
scenes = InternVideo.load_scenes("/content/drive/MyDrive/datasets/AIC2023/Video/Videos_L09/L09_V001.mp4", "/content/drive/MyDrive/data1/AIC2023/L09_scenes_txt/L09_V001.txt").cuda()
model = InternVideo.load_model("./models/InternVideo-MM-L-14.ckpt").cuda()
text = InternVideo.tokenize(text_cand).cuda()

with torch.no_grad():
    text_features = model.encode_text(text)
    text_features = torch.nn.functional.normalize(text_features, dim=1)
    for i, scene in enumerate(scenes):
        scene = scene.cuda()
        video_features = model.encode_video(scene.unsqueeze(0))
        # Normalize
        video_features = torch.nn.functional.normalize(video_features, dim=1)
        print(f"scene {i} encoded")
        print(video_features.shape)

    # index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(video_features)
        # Check and convert to cpu index before writing

    if index_gpu_flag:
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, bin_path)
    # print("features written")

    # t = model.logit_scale.exp()
    # probs = (video_features @ text_features.T * t).softmax(dim=-1).cpu().numpy()

# print("Label probs: ")  # [[9.5619422e-01 4.3805469e-02 2.0393253e-07]]
# for t, p in zip(text_cand, probs[0]):
    # print("{:30s}: {:.4f}".format(t, p))