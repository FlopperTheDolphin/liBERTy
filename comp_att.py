import torch
import transformers
import numpy as np
import pandas as pd

from vs_constants import *
from loader import *
from scipy.special import kl_div
from scipy.spatial.distance import jensenshannon

def comp_matrix(tokenizer,model,sentence):
 print('> '+ MTX_CAL)
# sentence,id_sent = load_sentence(sent_path)
 e=tokenizer.encode(sentence, add_special_tokens=True)
 output=model(torch.tensor([e]))
 #Matrix to load
 attentions = output[3]
 #tokens from sentence
 tokens = tokenizer.convert_ids_to_tokens(e)
 print('> '+ MTX_COMP)
  
 return attentions,tokens
 
def save_matrix(dir_path,tokens,attentions,verbose = True):
 # dimension (12,1,12,n,n)
 print('> ' + MTX_SAVE)
 dic_max=dict()
 for i in range(12):
  for j in range(12):
   np_attention = attentions[i][0][j].detach().numpy()
   df = pd.DataFrame(data=np_attention,columns=tokens)
   file_name = dir_path+"/"+ "att-mtx_layer-"+str(i+1)+"_head-"+str(j+1)+".csv"
   df.to_csv(file_name, index=False)
   dic_max[str((i+1,j+1))] = get_att_max(df)+0
   if(verbose) :
    print('> [' + file_name + '] Saved')
 print('> '+ MTX_SAVE_COMP)
 save_in_json(dic_max,dir_path+"/max.json")

def select_sub_matrix_for_token(out_dir,id_sent,layer,head,token):
 mtx = load_matrix(out_dir,id_sent,layer,head)
 fram = mtx[token]
 
 return fram
 
def comp_token(tokenizer,sentence):
 e=tokenizer.encode(sentence, add_special_tokens=True)
 return tokenizer.convert_ids_to_tokens(e)

def tokens_to_sentence(l1,l2,path_cache,dic = None,old_sen=None):
 
 if dic == None:
  try:
   dic = load_from_json(path_cache)
   return dic
  except Exception:  
   dic = dict()
 
 if len(l2)==0 or len(l1)==0:
  dic['[SEP]']='[SEP]'
  return dic
  save_in_json(dic,path_cache)
 
 s = l1.pop(0)
 
 if(s == '[CLS]'):
  dic[s] = '[CLS]'
  save_in_json(dic,path_cache) 
  tokens_to_sentence(l1,l2,path_cache,dic,old_sen)
  return dic
  
 if(s == '[SEP]'):
  dic[s] = '[SEP]'
  tokens_to_sentence(l1,l2,path_cache,dic,old_sen)
  save_in_json(dic,path_cache) 
  return dic 
 
 
 if '##' in  s:
  dic[s] = old_sen
  l2.pop(0)
  tokens_to_sentence(l1,l2,path_cache,dic,old_sen)
  save_in_json(dic,path_cache) 
  return dic
 
 obj_v = l2.pop(0)
 v = str(obj_v)
 
 k=list()
 k.append(s)
 
 print('v: '+str(v))
   
 while(v != s and len(l1) != 0):
  t = l1.pop(0)
  print('t: '+str(t))
  k.append(t)
  t= t.replace('##','') 
  s = s+t
  print('s: '+str(s))
   
 for token in k:
  dic[token] = v 
   
 old_sen = v
   
 tokens_to_sentence(l1,l2,path_cache,dic,old_sen)
 return dic   

 
def labelizer(l1,l2,path_cache,dic_token=None,old_pos=None):

 if dic_token == None:
  try:
   dic_token = load_from_json(path_cache)
   return dic_token
  except Exception:  
   dic_token = dict()
  
 if len(l2)==0 or len(l1)==0:
  dic_token['[SEP]']='[SEP]'
  save_in_json(dic_token,path_cache)
  return dic_token
 
 s= l1.pop(0)
 
 if(s == '[CLS]'):
  dic_token[s] = '[CLS]' 
  labelizer(l1,l2,path_cache,dic_token,old_pos)
  save_in_json(dic_token,path_cache)
  return dic_token
  
 if(s == '[SEP]'):
  dic_token[s] = '[SEP]'
  labelizer(l1,l2,path_cache,dic_token,old_pos)
  save_in_json(dic_token,path_cache)
  return dic_token 
   
 if '##' in  s:
  dic_token[s] = old_pos
  l2.pop(0)
  print('## trovato in: ' + str(s))
  labelizer(l1,l2,path_cache,dic_token,old_pos)
  save_in_json(dic_token,path_cache)
  return dic_token
 
 k=list()
 k.append(s)
 
 obj_v = l2.pop(0)
 v = str(obj_v)
 #obj_v troviamo tutto l'oggetto ma a noi ci serve anche la sua stringa
 #per il confronto
 
 #print('v no in ciclo : '  +str(v))
 while(v != s and len(l1) != 0):
#  print('v: ' + str(v))  
  t = l1.pop(0)#.pop(0)
  #print('t: ' + str(t))
  k.append(t)
  t= t.replace('##','')
  s=s+t 
 # print('s:' + str(s))
  
 old_pos = obj_v.pos_
 
 for token in k:
  dic_token[token] = obj_v.pos_ 
   
 labelizer(l1,l2,path_cache,dic_token,old_pos)
 save_in_json(dic_token,path_cache)
 return dic_token   

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

def get_att_max(att_mtx):
 return att_mtx.max().max()
    
def get_all_att_sentece(out_dir,sent_id,token):
 l = dict()
 for i in range(12):
  for j in range(12):
   l[str((i+1,j+1))] = select_sub_matrix_for_token(out_dir,sent_id,str(i+1),str(j+1),token)
 return l  
 
def comp_divergence(dic_att,n_token):
 # find for all attentions Series given a token, the divergence distance from
 # an average probability. 
 index_list=list()
 div_list=list()
 b_prob = np.array([1/n_token]*n_token)
 df = pd.DataFrame(columns=['divergence','head'])
 for i in range(12):
  for j in range(12):
   index_list.append(str((i+1,j+1)))
   a_prob = np.array(dic_att[str((i+1,j+1))])
   div = jensenshannon(a_prob, b_prob)
   div_list.append(div)
 
 df=pd.DataFrame(data=div_list,index=index_list,columns=['divergence'])    
  
 return df,index_list
  
 
   
