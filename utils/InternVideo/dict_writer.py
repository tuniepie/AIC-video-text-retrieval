import glob
import json
import re
import numpy as np

def indexing(root_scenes_dir, des_scenes_dict):
    infos = []
    txt_files = sorted(glob.glob(f"{root_scenes_dir}/**/*.txt", recursive=True))
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            list_scenes = f.readlines()
        list_scenes =  np.array([re.sub('\[|\]', '', line).strip().split(' ') for line in list_scenes]).astype(np.uint32)
        # base on start numbers and end numbers from txt files to load scenes from videos
        video_name = txt_file.split('/')[-1].replace('.txt', '')
        # start, end from scene txt lines
        for start, end in list_scenes:
            # print(f"video: {video_name}, start: {start}, end: {end}")
            info = {
                "video": video_name,
                "start": int(start),
                "end": int(end)
            }
            infos.append(info)
    # print(infos)
            
    scenes_dict = dict(enumerate(infos))
    print("turned infos to dict")
    with open(des_scenes_dict, 'w') as f:
        f.write(json.dumps(scenes_dict))
    print("done")
    
def main():
    root = '/home/hoangtv/Desktop/Attention/lst_scenes'
    des = '/home/hoangtv/Desktop/Attention/scenes_dict_test.json'
    
    indexing(root_scenes_dir=root, des_scenes_dict=des)

if __name__ == "__main__":
    main()