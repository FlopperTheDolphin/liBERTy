import os
#import spacy
from fun.vs_constants import *
from fun.loader import load_from_json,save_in_json
from fun.view import console_show,view_find
from fun.loader import load_tokenizer
from fun.comp_att import comp_token
 

def get_sentence(out_dir,name):
 
 dic_sent = load_from_json(os.path.join(os.path.join(out_dir, name),'sentence.json'))
 return dic_sent['sentence']

def get_bert_and_spacy_tokens(model_dir,sentence,mtx_dir):
 import spacy
 console_show(MSG_TOKENS_COMP)
 
 bert_tokens = get_bert_tokens(mtx_dir,model_dir,sentence)
 
 spacy_path = os.path.join(mtx_dir,"spacy_tokens.json")
 
 if not os.path.exists(spacy_path):  
  nlp = spacy.load("it_core_news_sm")
  doc = nlp(sentence)
  spacy_tokens=list()
  for token in doc:
   spacy_tokens.append(str(token)) 
  console_show(MSG_TOKEN_COMP_END)
  save_in_json(spacy_tokens,spacy_path)
 else:
  spacy_tokens=load_from_json(spacy_path) 
 
 return bert_tokens,spacy_tokens 


def tokens(name,out_dir,model_dir,view=True):
 sentence = get_sentence(out_dir,name)
 bert_tokens = get_bert_tokens(os.path.join(out_dir,name),model_dir,sentence)
 if view==True:
  console_show('',bert_tokens,pick=False)
 return bert_tokens


def sentence(out_dir,name):
 sentence = get_sentence(out_dir,name)
 console_show('',sentence,pick=False)
 
def get_bert_tokens(mtx_dir,model_dir,sentence):
 bert_path = os.path.join(mtx_dir,"bert_tokens.json")
 
 if not os.path.exists(bert_path): 
  tokenizer =  load_tokenizer(model_dir)
  bert_tokens = comp_token(tokenizer,sentence)
  save_in_json(bert_tokens,bert_path)
 else:
  bert_tokens=load_from_json(bert_path)
  
 return bert_tokens  
 
def find(sentence,bert_tokens,token,view=True,repeat=True):
  possible_index=list()
  j=0
  for tk in bert_tokens:
   if tk == token:
    possible_index.append(j)
    j=j+1
    
  if len(possible_index) == 0 and repeat == True:
   return find(sentence,bert_tokens,'##' + token,view,False)  
      
  if view== True:
   view_find(bert_tokens,token,possible_index)
   
  return possible_index  
   
