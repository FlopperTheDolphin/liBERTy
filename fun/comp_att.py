import torch
import transformers
import numpy as np
import pandas as pd
from fun.loader import load_matrix,save_in_json,load_from_json
from fun.vs_constants import *
from scipy.special import kl_div
from scipy.spatial.distance import jensenshannon
from scipy.interpolate import interp1d
from scipy.stats import norm

def comp_matrix(tokenizer,model,sentence):
 e=tokenizer.encode(sentence, add_special_tokens=True)
 output=model(torch.tensor([e]))
 attentions = output[3]
 tokens = tokenizer.convert_ids_to_tokens(e) 
 return attentions,tokens
 

#def select_sub_matrix_for_token(out_dir,id_sent,layer,head,token):
 #mtx = load_matrix(out_dir,id_sent,layer,head)
 #try:
  #fram = mtx[token]
 #except Exception:
  #fram = mtx['##'+token] 
 
 #return fram


def select_sub_matrix_for_token(out_dir,id_sent,layer,head,token):
 mtx = load_matrix(out_dir,id_sent,layer,head)
 sel=True
 ha = ''
 frams = list()
 try:
   frams.append(mtx[token])
 except Exception:
   token = '##'+token
   frams.append(mtx[token])
   ha='##'
    
 j=1
 t = ''+token
 token = t+'.'+str(j)
 while(sel):
  try:
   frams.append(mtx[token])
   j=j+1
   token = t+'.'+str(j)
  except Exception:
   sel=False

 return frams,j,ha

def update_token(j,has,token):
 if j == 0:
  return str(has) + token
 else: 
  return str(has) + token +'.'+str(j)
 
def comp_token(tokenizer,sentence):
 e=tokenizer.encode(sentence, add_special_tokens=True)
 return tokenizer.convert_ids_to_tokens(e)

def tokens_to_sentence(l1,l2,dic = None,old_sen=None):
 
 if dic == None:  
   dic = dict()
 
 if len(l2)==0 or len(l1)==0:
  dic['[SEP]']='[SEP]'
  return dic
 
 s = l1.pop(0)
 
 if(s == '[CLS]'):
  dic[s] = '[CLS]'
  tokens_to_sentence(l1,l2,dic,old_sen)
  return dic
   
 if(s == '[SEP]'):
  dic[s] = '[SEP]'
  tokens_to_sentence(l1,l2,dic,old_sen) 
  return dic 
 
 
 if '##' in  s:
  dic[s] = old_sen
  l2.pop(0)
  tokens_to_sentence(l1,l2,dic,old_sen)
  return dic
 
 obj_v = l2.pop(0)
 v = str(obj_v)
 
 k=list()
 k.append(s)
   
 while(v != s and len(l1) != 0):
  t = l1.pop(0)
  
  k.append(t)
  t= t.replace('##','') 
  s = s+t
 
   
 for token in k:
  dic[token] = v 
   
 old_sen = v
   
 tokens_to_sentence(l1,l2,dic,old_sen)
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
    
def get_all_att_sentece(out_dir,sent_id,token,id_token=0):
 l = dict()
 for i in range(12):
  for j in range(12):
   frams,k,has = select_sub_matrix_for_token(out_dir,sent_id,str(i+1),str(j+1),token)
   l[str((i+1,j+1))] = frams[int(id_token)]
 return l  
 
def comp_divergence(dic_att,n_token,bert_tokens,weight='all'):
 # find for all attentions Series given a token, the divergence distance from
 # an average probability. 
 index_list=list()
 div_list=list()
 if weight == 'all':
  b_prob = np.array([1/n_token]*n_token)
 else:
  j=0
  for i in range(n_token):
   if bert_tokens[i] in weight: 
     j=j+1
     
  w = 1/j
  b_prob = np.array([0]*n_token,dtype=float)
  for i in range(n_token):
   if bert_tokens[i] in weight:
    b_prob[i] = w
 
 A = get_head_matrix()
 df = pd.DataFrame(columns=['divergence','head'])
 for i in range(12):
  for j in range(12):
   index_list.append(str((i+1,j+1)))
   a_prob = np.array(dic_att[str((i+1,j+1))])
   div = jensenshannon(a_prob, b_prob)
   A[(i,j)] = div
   div_list.append(div)
 
 df=pd.DataFrame(data=div_list,index=index_list,columns=['divergence'])    
  
 return df,index_list,A
  
def comp_token_weight(tokens,layer,head,name,out_dir):
  l=list()
  for tk in tokens:
   s = select_sub_matrix_for_token(out_dir,name,layer,head,tk).sum(axis=0)
   l.append(s)
   
  return l

def get_all_matrix(out_dir,name):
 dic_head = dict()
 l_index = list()
 for i in range(12):
  for j in range(12):
   mtx = load_matrix(out_dir,name,str(i+1),str(j+1))
   head_index = str((i+1,j+1))
   dic_head[head_index] = mtx
   l_index.append(head_index)
 return dic_head,l_index         

 
def get_ordered_token(att_token,bert_tokens):
 att_token = (att_token/att_token.max())*100 
 token_attention = list()
 for i in range(len(bert_tokens)):
  token_attention.append((bert_tokens[i],att_token[i]))

 return pd.DataFrame(data=token_attention,columns=['token','attention']).sort_values(by='attention',ascending=False)
 
def get_smear(fram,pos,bert_tokens):
 att_tokens = fram['attention'][pos]
 most_important_token = list(fram.loc[fram['attention'] > 5].index)
 
 smear = sm(pos+1,most_important_token,'r') + sm(pos-1,most_important_token,'l')
 
 for s in smear:
  if s in most_important_token:
   most_important_token.remove(s)
   
 return smear,most_important_token   

def sm(next,most_important,direction,smear=None):

 if smear == None:
  smear = list()
  
 if next < 0:
  return smear
 
 if next in most_important:
  smear.append(next)
  if direction == 'r':
   next=next+1
   smear = sm(next,most_important,direction,smear)
   return smear
  elif direction == 'l':
   next=next-1
   smear = sm(next,most_important,direction,smear)
   return smear
  
 else:
  return smear
 
def get_index_from_token(token,j,has,bert_tokens,frams):
 ks = list()
 i=0
 for k in range(len(bert_tokens)):
  if bert_tokens[k] == token: 
   ks.append((frams[i],k))
   i=i+1
 return ks
   
def comp_avg_and_std(div):
 return np.mean(div),np.std(div)   
 
def update_matrix(A,layer,head,a):
   A[(layer,head)] = a
   return A

def get_head_matrix():
 return np.zeros((12,12))

def get_max(A):
 return np.amax(A)   
 
def sort_outliers(l_diff,bert_tokens):
 return pd.DataFrame(data=l_diff,index=bert_tokens,columns=['att_diff']).sort_values(by='att_diff',ascending=False)
  
  
