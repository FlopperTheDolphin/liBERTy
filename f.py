from comp_att import * #save_matrix,comp_matrix,comp_token,select_sub_matrix_for_token,load_matrix
from loader import * #save_sentence,save_matrix,load_tokenizer
from view import * #console_show,  view_attention_gradient,view_top_tokens,view_higher_token
from vs_constants import * #MSF_MDL_LOADING,MSG_MDL_LOADED,MSG_MTX_DIR
import sys
import os
#import getopt
import gc
import spacy

###### LOAD SENTENCE MATRIX AND CREATE ENVIORMENT ##########################

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
############################################################################################ 

#def see_total(name,layer,head):
 # dic_sent = load_from_json(os.path.join(os.path.join(out_dir, name),'sentence.json'))
 # sentence = dic_sent['sentence']
 # tokenizer =  load_tokenizer(model_dir)
 # tokens = comp_token(tokenizer,sentence)
  
 # list_weight=comp_token_weight(tokens,layer,head,name,out_dir)
  
 # view_att_total(name,list_weight,tokens)


############## SEE ATTENTION STAT FOR A SENTENCE #################################################
def see_attention_sentence(name,layer,head,word,sent,cls,out_dir,model_dir,time=None):

  sentence = get_sentence(out_dir,name)  
  mtx_dir = create_token_path_file(out_dir,name)
  bert_tokens,spacy_tokens = get_bert_and_spacy_tokens(model_dir,sentence,mtx_dir) 
  dic_sent = unify_bert_and_spacy_tokens(spacy_tokens,bert_tokens)
 
  for token in bert_tokens:
   if dic_sent[token] == word:
    fram = select_sub_matrix_for_token(out_dir,name,layer,head,token) 
    if sent == True:
     plot_gradient_attention(mtx_dir,fram,bert_tokens,token,layer,head) 
     if time != None: 
      view_top_tokens(fram,bert_tokens,token,time)
    if cls == True: 
     view_higher_token(fram,bert_tokens,token,time)  
 
def create_token_path_file(out_dir,name):
 mtx_dir = os.path.join(out_dir, name)
 return mtx_dir  
  
def unify_bert_and_spacy_tokens(spacy_tokens,bert_tokens):
 console_show(MSG_UNIFY_TOKEN)
 sp = list()+spacy_tokens 
 bt = list()+bert_tokens 
 dic_sent = tokens_to_sentence(bt,sp)
 console_show(MSG_UNIFY_TOKEN_END)
 return dic_sent
 
def plot_gradient_attention(mtx_dir,fram,bert_tokens,token,layer,head):
  max_path= os.path.join(mtx_dir,"max.json")
  mx= load_from_json(max_path)
  view_attention_gradient(fram,bert_tokens,token,layer,head,mx)      
  
###############################################################################################


########################### FIND SHANNON DIVERGENCE IN ALL THE MATRIX GIVEN A TOKEN ###################
def see_stat(name,out_dir,token,model_dir):
  sentence,bert_tokens = get_sentence_and_bert_tokens(out_dir,name,model_dir)
  dic_att=from_token_select_all_columns(out_dir,name,token)
  n_token = len(bert_tokens)
  df,l=comp_divergence(dic_att,n_token)
  view_token_div(df,token,l)
   

def comp_stat(name,out_dir,perc,model_dir):

  sentence,bert_tokens = get_sentence_and_bert_tokens(out_dir,name,model_dir)
  mn,mx,freq_path,mat_token_path=initialise_perc_and_define_paths(out_dir,name,perc)
      
  if not os.path.exists(freq_path) or not os.path.exists(mat_token_path):
    dic_head,dic_token,n_token=get_dictionaries_and_length(bert_tokens)
    for token in bert_tokens:
     console_show(TOKEN,token)
     heads_chosen,df,l = select_head_by_mn_mx(out_dir,name,token,n_token,mn,mx)
     dic_head,dic_token=create_file_freq_and_save(heads_chosen,dic_head,l,dic_token,token,freq_path,mat_token_path)
  else:
    dic_token = load_from_json(mat_token_path)
    dic_head = load_from_json(freq_path)
    
  view_chosen_heads(dic_head,dic_token,mn,mx,bert_tokens) 
    
  # Parte aggiunta per controllare se per caso in ogni token esistesse qualche informazione tra le head trovate
  # infatti sono state prese in considerazione le head con una frequenza minore nel calcolo max di divergenze di shannon
  # purtroppo niente di nuovo, anzi le head hanno mostrato fenomeni di offset singolari in quelle head isolate


def get_sentence_and_bert_tokens(out_dir,name,model_dir):
  sentence = get_sentence(out_dir,name)
  mtx_dir = create_token_path_file(out_dir,name)
  bert_tokens=get_bert_tokens(mtx_dir,model_dir,sentence)
  return sentence,bert_tokens

def select_head_by_mn_mx(out_dir,name,token,n_token,mn,mx):
  dic_att = get_all_att_sentece(out_dir,name,token) 
  df,l=comp_divergence(dic_att,n_token) 
  heads_chosen = list(df.loc[(df['divergence'] >= (int(mn)/100)) & (df['divergence'] <= (int(mx)/100))].index.values)
  console_show(MSG_HEAD_CHOSEN,heads_chosen)
  return heads_chosen,df,l
  
def create_file_freq_and_save(heads_chosen,dic_head,l,dic_token,token,freq_path,token_path):  
  for head in heads_chosen:
   if head in dic_head.keys():
    dic_head[head] = dic_head[head] + 1
   else:
    dic_head[head] = 1
   # index_l=l       
  dic_token[token] = heads_chosen
    
  save_in_json(dic_head,freq_path)
  save_in_json(dic_token,token_path)
  return dic_head,dic_token
  
  
def initialise_perc_and_define_paths(out_dir,name,perc): 
   mn = perc   
   mx = int(perc)+10 
   freq_path=os.path.join(os.path.join(out_dir,name),"head_offset_"+str(perc)+".json")
   mx_token_path=os.path.join(os.path.join(out_dir,name),"token_offset_"+str(perc)+".json")
   return mn,mx,freq_path,mx_token_path

def get_dictionaries_and_length(tokens):
   dic_head = dict() #lol
   dic_token = dict()
   n_token = len(tokens)
   return dic_head,dic_token,n_token

def get_bert_tokens(mtx_dir,model_dir,sentence):
 bert_path =  max_path= os.path.join(mtx_dir,"bert_tokens.json")
 
 if not os.path.exists(bert_path): 
  tokenizer =  load_tokenizer(model_dir)
  bert_tokens = comp_token(tokenizer,sentence)
  save_in_json(bert_tokens,bert_path)
 else:
  bert_tokens=load_from_json(bert_path)
  
 return bert_tokens  
 
def from_token_select_all_columns(out_dir,name,token):
 console_show(MSG_SENT,name)
 console_show(MSG_COMP_COLUMN)
 dic_att= get_all_att_sentece(out_dir,name,token) 
 console_show(MSG_COMP_COLUMN_END)
 return dic_att
 
def who(name,layer,head,perc,out_dir,model_dir):
 sentence,bert_tokens = get_sentence_and_bert_tokens(out_dir,name,model_dir)
 token_path=os.path.join(os.path.join(out_dir,name),"token_offset_"+str(perc)+".json")  
 if check_file_exists(token_path) != True:
  return
 dic_token = load_from_json(token_path)   
 view_chosen_tokens(layer,head,bert_tokens,dic_token)
 
def check_file_exists(token_path):
 if not os.path.exists(token_path):
  console_show(MSG_FILE_NOT_FOUND)
  return False
 else:
  return True 
   
########################################################################################################################

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

def tokens(name,out_dir,model_dir):
 sentence,bert_tokens = get_sentence_and_bert_tokens(out_dir,name,model_dir)
 console_show('',bert_tokens,pick=False)

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
  
