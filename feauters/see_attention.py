import os
from fun.vs_constants import *
from fun.loader import load_from_json
from feauters.utiliy import get_sentence,get_bert_and_spacy_tokens
from fun.comp_att import tokens_to_sentence,select_sub_matrix_for_token
from fun.view import console_show,view_attention_gradient,view_top_tokens,view_higher_token

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

