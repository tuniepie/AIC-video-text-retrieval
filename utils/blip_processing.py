# import faiss
# import requests
# from torchvision import transforms
# from torchvision.transforms.functional import InterpolationMode
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# from nltk.corpus import stopwords
from blip_models.blip_itm import blip_itm
from translate_processing import Translation
# from nlp_processing import Translation, Text_Preprocessing
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import math
import os, sys
import csv

FILE = Path(__file__).resolve()
# Read folder containing file path
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
# main work directory
WORK_DIR = os.path.dirname(ROOT)
    
mode_compute = 'blip'
# for keyframes dictionary
data_path = os.path.join(WORK_DIR, 'data')
keyframes_id_path = os.path.join(data_path, 'dict/keyframes_id_path.json')
stopword_dict_path = os.path.join(data_path, 'dict/vietnamese-stopwords-dash.txt')
# for features file
folder_features = os.path.join(data_path, 'bins')
bin_path = os.path.join(folder_features, f'{mode_compute}_faiss_cosine.bin')
# for results
result_path = os.path.join(WORK_DIR, 'results')
mode_result_path = os.path.join(result_path, mode_compute)


def show_images(image_paths, scores, timestamp, save_path='results', method='text'):
    fig = plt.figure(figsize=(15, 10))
    columns = int(math.sqrt(len(image_paths)))
    rows = int(np.ceil(len(image_paths)/columns))
    

    for i in range(columns*rows):

        img = plt.imread(image_paths[i])
        ax = fig.add_subplot(rows, columns, i+1)
        ax.set_title('/'.join(image_paths[i].split('/')[-2:]).replace('/',',').replace('.jpg',f' {i}'))
        # ax.set_title('/'.join(image_paths[i].split('/')[-2:]) + str(scores[:, i]))

        plt.imshow(img)
        plt.axis("off")
        plt.savefig(os.path.join(save_path, f'{timestamp}_{method}_retrieval.jpg'))
    plt.show()
    
def format_for_CSV(image_paths, id):
    shortened_path = []  # Danh sách kết quả

    for i, path in enumerate(image_paths):
        parts = path.split("/")  # Tách đường dẫn thành các phần
        folder_name = parts[-3]  # Tên thư mục chứa loại và phiên bản
        file_name = parts[-1].split(".")[0]  # Tên tập tin mà không có phần mở rộng
        shortened = f"{folder_name}, {file_name} {i} "
        shortened_path.append(shortened)
    
    return shortened_path


def write_results(image_paths, input_ids,output_path):
    shortened_path = []  # Danh sách kết quả
    
    for i, path in enumerate(image_paths):
        parts = path.split("/")  # Tách đường dẫn thành các phần
        folder_name = parts[-2]  # Tên thư mục chứa loại và phiên bản
        file_name = parts[-1].split(".")[0]  # Tên tập tin mà không có phần mở rộng
        shortened = f"{folder_name}, {file_name}"
        shortened_path.append(shortened)

    # In ra danh sách kết quả
    # print(shortened_path)

    # ID bạn muốn chọn
    

    # Chuyển đổi chuỗi nhập vào thành danh sách các ID (kiểu int)
    selected_ids = [int(id.strip()) for id in input_ids.split(' ')]

    # Tạo danh sách các shortened path được chọn
    selected_paths = []
    for selected_id in selected_ids:
        for i, path in enumerate(shortened_path):
        # Trích xuất ID từ shortened path
            if i == selected_id:
                selected_paths.append(path)  # Lấy shortened path nếu ID trùng khớp

    # Kiểm tra xem có shortened path được chọn không
    if selected_paths:
        # Ghi vào tệp CSV
        output_file = os.path.join(output_path,'selected_shortened_paths.csv')
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
        
            for path in selected_paths:
                writer.writerow([path])

        print(f'Các shortened path đã được chọn đã được ghi vào tệp {output_file}')
    else:
        print(f'Không tìm thấy shortened path với các ID đã chọn')
        

def stopwords_filtering(text):
    # stopwords_list = stopwords.words('english')
    words = [word for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS]
    new_text = " ".join(words)
    return new_text 


# Note: before running, cd to '/home/hoangtv/Desktop/Attention/txt2vid_ret'
def main():

    if os.getcwd() != WORK_DIR:
        print("Changing to proper working directory...")
        os.chdir(WORK_DIR)
        print(f"Done, working directory: {os.getcwd()}")

    if not os.path.exists(os.path.join(result_path)):
        os.makedirs(result_path)

    if not os.path.exists(os.path.join(mode_result_path)):
        os.makedirs(mode_result_path)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    image_size = 384
    
    translator = Translation()

    # model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
    model_path = os.path.join(WORK_DIR, 'models/model_base_retrieval_coco.pth')
        
    ## Model = BLIP search
    model = blip_itm(pretrained=model_path, image_size=image_size, vit='base', keyframes_dict=keyframes_id_path, features_path=bin_path)
    model.eval()
    model = model.to(device)
    model.load_index_from_bin_file()
    print(f"init model & index done, n.o. features: {model.index.ntotal}")
    
    while True:
        text_input = input("Enter your query: ")
        translated_text = translator(text_input)
        filtered_text = stopwords_filtering(translated_text)
        print(f"Query text (translated & filtered): {filtered_text}")

        timer = time.time()
        scores, images_id, image_paths = model.text_search(translated_text, k=20)
        # select_input = input("Enter your selection: ")
        # print(scores)
        # print(images_id)
        # print(image_paths)
        print(f"Time: {time.time() - timer}")

        # print(image_paths)
        # show_images(image_paths=image_paths, scores=scores, timestamp=timer)
        show_images(image_paths=image_paths, scores=scores, timestamp=timer, save_path=mode_result_path)
        input_ids = input("Nhập các ID bạn muốn chọn (cách nhau khoảng trắng): ")
        if input_ids == "": 
            continue
        else: write_results(image_paths, input_ids,mode_result_path)
        print('======== End of Program =========')

if __name__ == "__main__":
    main()