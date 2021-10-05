import os
from feauters.utiliy import get_sentence,get_bert_and_spacy_tokens 
from fun.vs_constants import *
from fun.loader import load_from_json,save_in_json,load_matrix
from fun.view import console_show,view_loaded_pos,view_mtx_pos
from fun.comp_att import labelizer,get_pos_mtx


def see_pos(name,layer,head,out_dir,model_dir):
  sentence = get_sentence(out_dir,name)

  save,path_img,path_graph,path_cache = define_enviorment(out_dir,name,layer,head)
  
  if(view_loaded_pos(path_graph,path_img,save) == True):
   return  
  
  bert_tokens,spacy_tokens = get_bert_and_spacy_tokens(model_dir,sentence,os.path.join(out_dir, name))
  dic_tokens = get_label(bert_tokens,spacy_tokens,path_cache)
  att_mtx = load_matrix(out_dir,name,layer,head)
  dic_pos,dic_edge = get_pos_mtx(att_mtx,dic_tokens,bert_tokens)
  graph_json = view_mtx_pos(dic_edge,dic_pos,path_cache,path_img,save)
  save_in_json(graph_json,path_graph)  

def define_enviorment(out_dir,name,layer,head):
 path_graph = os.path.join(os.path.join(out_dir,name),"pos-graph_layer-"+str(layer)+"_head-"+str(head)+".json")
  
 if not os.path.exists(path_graph):
   os.mknod(path_graph)
  
 path_img = os.path.join(os.path.join(out_dir, name),"img-graph_layer-"+str(layer)+"_head-"+str(head)+".png")
 
 path_cache = os.path.join(os.path.join(out_dir, name),"token_pos.json") 
  
 if not os.path.exists(path_img):
  save = True
 else:
  save = False
 return save,path_img,path_graph,path_cache 
  
def get_label(bert_tokens,spacy_tokens,path_cache):
 bt = list() + bert_tokens
  
 if not os.path.exists(path_cache):
   os.mknod(path_cache)
  
 return labelizer(bt,spacy_tokens,path_cache)
  
