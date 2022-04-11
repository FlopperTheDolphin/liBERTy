import os
from constants import *
from fun.view import console_show
from fun.loader import load_model,save_sentence,save_matrix,save_hidden_state,loading_tf_model,save_score,save_one_matrix,load_matrix
from fun.comp_att import comp_matrix,comp_score
from features.utiliy import get_bert_tokens



def load(sentence,sent_id,name,model_dir,out_dir): 
 tokenizer,model = loading_model_and_tokenizer(model_dir)
 attentions,tokens,hidden = comp_mtx(tokenizer,model,sentence)
 mtx_path,max_path,sent_path = create_enviorment_and_path(out_dir,name)
 save_sentence(sentence,sent_id,sent_path)
 save_hidden_state(mtx_path,hidden,tokens)
 att_max = save_in_file_attention_matrix(mtx_path,tokens,attentions)
 

def load_score(model_dir,out_dir,name,sentence):
 model = loading_tf_model(model_dir)
 dic_score = comp_score(model)
 mtx_path,max_path,sent_path = create_enviorment_and_path(out_dir,name)
 save_score(dic_score,mtx_path)
 console_show(SCORE_COM) 
 tokens =get_bert_tokens(os.path.join(out_dir, name),model_dir,sentence)
 import numpy as np
 for i in range(12):
  sm=0
  mtx_sum = np.zeros([len(tokens),len(tokens)])
  for j in range(12):
   score = dic_score[str((i+1,j+1))]
   mtx = load_matrix(out_dir,str(name),str(i+1),str(j+1)).to_numpy()
   digest_mtx = mtx*score
   sm=sm+score 
   mtx_sum = mtx_sum + digest_mtx
  layer_mtx = mtx_sum/sm
  save_one_matrix(layer_mtx,tokens,'layer-'+str(i+1),os.path.join(out_dir, name))
   
   
def comp_mtx(tokenizer,model,sentence):
 console_show(MSG_MTX_CAL)
 attentions,tokens,hidden = comp_matrix(tokenizer,model,sentence)
 console_show(MSG_MTX_COMP)
 return attentions,tokens,hidden

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

