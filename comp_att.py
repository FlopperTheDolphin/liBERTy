import torch
import transformers
import numpy as np
import pandas as pd

from vs_constants import *
from loader import *

def comp_matrix(tokenizer,model,sent_path):
 print('> '+ MTX_CAL)
 sentence,id_sent = load_sentence(sent_path)
 e=tokenizer.encode(sentence, add_special_tokens=True)
 output=model(torch.tensor([e]))
 #Matrix to load
 attentions = output[3]
 #tokens from sentence
 tokens = tokenizer.convert_ids_to_tokens(e)
 print('> '+ MTX_COMP)
  
 return attentions,tokens,id_sent
 
def save_matrix(dir_path,tokens,attentions,verbose = True):
 # dimension (12,1,12,n,n)
 print('> ' + MTX_SAVE)
 for i in range(12):
  for j in range(12):
   np_attention = attentions[i][0][j].detach().numpy()
   df = pd.DataFrame(data=np_attention,columns=tokens)
   file_name = dir_path+"/"+ "att-mtx_layer-"+str(i+1)+"_head-"+str(j+1)+".csv"
   df.to_csv(file_name, index=False)
   if(verbose) :
    print('> [' + file_name + '] Saved')
 print('> '+ MTX_SAVE_COMP)   


def select_sub_matrix_for_token(out_dir,id_sent,layer,head,token):
 mtx = load_matrix(out_dir,id_sent,layer,head)
 fram = mtx[token]
 
 return fram
 
def comp_token(tokenizer,sent_path):
 sentence,id_sent = load_sentence(sent_path)
 e=tokenizer.encode(sentence, add_special_tokens=True)
 return tokenizer.convert_ids_to_tokens(e)
 
def labelizer(l1,l2,dic=None):
 if dic == None:
  dic = dict()
  
 if len(l2)==0 or len(l1)==0:
  dic['[SEP]']='[SEP]'
  return dic
 
 s = l1.pop(0)
 
 if(s == '[CLS]'):
  dic[s] = '[CLS]' 
  labelizer(l1,l2,dic)
  return dic
  
 if(s == '[SEP]'):
  dic[s] = '[SEP]'
  labelizer(l1,l2,dic)
  return dic 
   
 obj_v = l2.pop(0)
 v = str(obj_v)
 #obj_v troviamo tutto l'oggetto ma a noi ci serve anche la sua stringa
 #per il confronto
 
 k=list()
 k.append(s)
 
 while(v != s and len(l1) != 0):
  t = l1.pop(0)
  k.append(t)
  t= t.replace('##','') 
  s = s+t 
 
 for token in k:
  dic[token] = obj_v.pos_ 
   
 labelizer(l1,l2,dic)
 return dic   

def update_pos(pos=None,dic=None):
 if pos == None:
  return dict()
  
 if dic == None:
  dic = dict()
  
 if pos not in dic.keys():
  dic[pos] = len(dic)
  
 return dic

def get_pos_mtx(att_mtx,dic_tokens,tokens):
   
   dic_pos = update_pos()
   dic_edge = dict()
   
      
   for i in range(len(tokens)):
    token1 = tokens[i]
  #  if(token1 != '[SEP]' and token1 !='[CLS]'):
    pos1 = dic_tokens[token1]
    dic_pos = update_pos(pos1,dic_pos)
    column = att_mtx[token1].tolist()
    for j in range(i+1):
     token2 = tokens[j]
     weight = column[j]
     pos2 = dic_tokens[token2]
     dic_pos = update_pos(pos2,dic_pos)
     
     if dic_pos[pos1]+dic_pos[pos2] not in dic_edge.keys():
      dic_edge[dic_pos[pos1]+dic_pos[pos2]] = weight
     else:
      dic_edge[dic_pos[pos1]+dic_pos[pos2]] = dic_edge[dic_pos[pos1]+dic_pos[pos2]] + weight    
         
   return dic_pos,dic_edge    
 
