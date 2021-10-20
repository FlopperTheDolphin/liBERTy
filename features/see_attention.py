import os
from fun.vs_constants import *
from fun.loader import load_from_json,save_in_json
from features.utiliy import get_sentence,get_bert_and_spacy_tokens,get_bert_tokens
from fun.comp_att import tokens_to_sentence,select_sub_matrix_for_token,get_head_matrix,get_ordered_token,get_smear,select_columns_for_token,update_token,get_index_from_token
from fun.view import console_show,view_attention_gradient,view_top_tokens,view_higher_token,view_smear,view_noop,view_matrix,view_total_noop

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

def smear(name,out_dir,model_dir,token,layer,head):
 #find normal distribution for all the datas
 sentence = get_sentence(out_dir,name)
 bert_tokens = get_bert_tokens(os.path.join(out_dir, name),model_dir,sentence)
 frams,j,has =select_columns_for_token(out_dir,name,layer,head,token)
 inds = get_index_from_token(token,j,has,bert_tokens,frams)
 for i in inds:
  fram = get_ordered_token(i[0],bert_tokens)
  smear,drops = get_smear(fram,i[1],bert_tokens)
  print(smear)
  print(drops)
  
 print('smear') 
 for s in smear:
  print(bert_tokens[s]) 
 print('')
 print('drops')
 for d in drops:
  print(bert_tokens[d]) 
 
 
 
 #view_smear(smear,drop,bert_tokens)
 
   
def noop_search(name,out_dir,model_dir,layer,head):
 bert_tokens = initialise_noop(name,out_dir,model_dir)
 noop_sel(out_dir,name,layer,head,bert_tokens)
 
def noop_sel(out_dir,name,layer,head,bert_tokens):
 dic_att = dict()
 for token in bert_tokens:
  dic_att[token] = select_sub_matrix_for_token(out_dir,name,layer,head,token).sum()
 
 return view_noop(dic_att,layer,head)
 

def initialise_noop(name,out_dir,model_dir):
 console_show(MSG_SENT,name)
 sentence = get_sentence(out_dir,name)
 bert_tokens = get_bert_tokens(os.path.join(out_dir, name),model_dir,sentence) 
 return bert_tokens
      
 
def total_noop(name,out_dir,model_dir):
 path_att = os.path.join(os.path.join(out_dir, name),'att_max.json') 
 bert_tokens = initialise_noop(name,out_dir,model_dir)
 if not os.path.exists(path_att):
  dic_att =dict()
  for i in range(12):
   for j in range(12):
    dic_att[str((i+1,j+1))] = noop_sel(out_dir,name,str(i+1),str(j+1),bert_tokens)
 
  save_in_json(dic_att,path_att)
 else:
  dic_att=load_from_json(path_att)
  
 A = get_head_matrix()    
   
 for i in range(12):
   for j in range(12):
    A[(i,j)] = dic_att[str((i+1,j+1))][1]
    
 view_total_noop(dic_att,A,len(bert_tokens))  

