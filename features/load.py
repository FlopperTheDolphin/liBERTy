import os
from fun.vs_constants import *
from fun.view import console_show
from fun.loader import load_model,save_sentence,save_matrix
from fun.comp_att import comp_matrix 

# save matrix in a file
def load(sentence,sent_id,name,model_dir,out_dir): 
 tokenizer,model = loading_model_and_tokenizer(model_dir)
 attentions,tokens = comp_mtx(tokenizer,model,sentence)
 mtx_path,max_path,sent_path = create_enviorment_and_path(out_dir,name)
 save_sentence(sentence,sent_id,sent_path)
 att_max = save_in_file_attention_matrix(mtx_path,tokens,attentions)
 
def comp_mtx(tokenizer,model,sentence):
 console_show(MSG_MTX_CAL)
 attentions,tokens = comp_matrix(tokenizer,model,sentence)
 console_show(MSG_MTX_COMP)
 return attentions,tokens

def save_in_file_attention_matrix(mtx_path,tokens,attentions):
  console_show(MSG_MTX_SAVE)
  att_max = save_matrix(mtx_path,tokens,attentions)
  console_show(MSG_MTX_SAVE_COMP)
  return att_max
     
def loading_model_and_tokenizer(model_dir):
 console_show(MSG_MDL_LOADING)
 tokenizer,model = load_model(model_dir,model_dir)
 console_show(MSG_MDL_LOADED)
 return tokenizer,model  
 
def create_enviorment_and_path(out_dir,name):
 mtx_path = os.path.join(out_dir, name)
 max_path= os.path.join(mtx_path,"max.json")
 sent_path= os.path.join(mtx_path,"sentence.json")
 
 if(not os.path.isdir(mtx_path)): 
  os.mkdir(mtx_path)
  console_show(MSG_MTX_DIR,mtx_path)
  
 return mtx_path,max_path,sent_path

