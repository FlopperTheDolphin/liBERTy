import os
from fun.vs_constants import *
from feauters.utiliy import get_sentence
from fun.loader import load_from_json,save_in_json
from fun.view import console_show,view_token_div,view_chosen_heads,view_chosen_tokens
from fun.comp_att import get_all_att_sentece,comp_divergence

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
  
def create_token_path_file(out_dir,name):
 mtx_dir = os.path.join(out_dir, name)
 return mtx_dir  

