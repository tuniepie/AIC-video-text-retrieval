from blip_models.med import BertConfig, BertModel
from transformers import BertTokenizer
# from translate_processing import Translation

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import faiss, faiss.contrib.torch_utils
import numpy as np
import json
import os
import pandas as pd
from pathlib import Path

from blip_models.blip import create_vit, init_tokenizer, load_checkpoint


FILE = Path(__file__).resolve()
# Read folder containing file path
ROOT = FILE.parents[1]
# main work directory
WORK_DIR = os.path.dirname(ROOT)


class BLIP_ITM(nn.Module):
    def __init__(self,                
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                      
                 embed_dim = 256,     
                 keyframes_dict = '/home/hoangtv/Desktop/Attention/txt2vid_ret/data/bins/blip_faiss_cosine.bin',
                 features_path = '/home/hoangtv/Desktop/Attention/txt2vid_ret/data/dict/keyframes_id_path.json'
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)          
        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.itm_head = nn.Linear(text_width, 2) 
        ###4 User add 
        self.index = None
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.keyframes_id = self.load_dict_from_json_file(keyframes_dict) # read keyframes_id.json
        self.features_path = features_path
        
        
    def forward(self, image, caption, match_head='itm'):

        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
      
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(image.device) 

                 
        if match_head=='itm':
            output = self.text_encoder(text.input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )
            itm_output = self.itm_head(output.last_hidden_state[:,0,:])     
            return itm_output
            
        elif match_head=='itc':
            text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')                     
            image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)   
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)    
            
            sim = image_feat @ text_feat.t()
            return sim
        

    def text_search(self, text, k):
        text_features = self.get_text_features(text, device=self.device).cpu().detach().numpy().astype(np.float32)
        scores, idx_image = self.index.search(text_features, k=k)
        idx_image = idx_image.flatten()
        image_paths = [self.keyframes_id[f"{str(i)}"] for i in idx_image]
        return scores, idx_image, image_paths

            
    def get_image_features(self, img_path, img_size, device):
        """Modified: now this function get img_path
        """
        raw_image = Image.open(img_path).convert('RGB')   
        # w,h = raw_image.size
        # display(raw_image.resize((w//5,h//5)))
        transform = transforms.Compose([
            transforms.Resize((img_size,img_size),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]) 
        image = transform(raw_image).unsqueeze(0).to(device)   
        image_embeds = self.visual_encoder(image) 
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)   
        return image_feat

        
    def get_text_features(self, caption, device):
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35,
                              return_tensors="pt").to(device)
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,
                                            return_dict = True, mode = 'text')
        # print(text_output.last_hidden_state.shape)
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)
        return text_feat


    def load_index_from_bin_file(self):
        bin_file = self.features_path
        self.index = faiss.read_index(bin_file)
        

    def load_dict_from_json_file(self, json_path: str):
        with open(json_path, 'r') as f:
            js = json.loads(f.read())
        return js

        
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
        
       
def blip_itm(pretrained='',**kwargs):
    model = BLIP_ITM(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model         
            