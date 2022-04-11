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
 hidden_states = output[2]
 #print(len(hidden_states))
 #print(hidden_states[0].shape)
 
 return attentions,tokens,hidden_states
 

#def select_sub_matrix_for_token(out_dir,id_sent,layer,head,token):
 #mtx = load_matrix(out_dir,id_sent,layer,head)
 #try:
  #fram = mtx[token]
 #except Exception:
  #fram = mtx['##'+token] 
 
 #return fram

def select_sub_matrix_for_index(out_dir,name,layer,head,ind_token):
 mtx = load_matrix(out_dir,name,layer,head)
 return mtx.iloc[[ind_token]]
 
 
def select_sub_matrix_for_token(out_dir,id_sent,layer,head,token,bert_tokens):
 mtx = load_matrix(out_dir,id_sent,layer,head)
 sel=True
 ha = ''
 frams = list()
 index=None
 
 index = sel_index_by_token(token,bert_tokens)
 if index == None:
  token = '##'+token
  ha='##'
  index = sel_index_by_token(token,bert_tokens)
 
 if index != None:
  frams.append(mtx.iloc[index]) 
  
 #try:
  # frams.append(mtx.iloc[i])#/mtx[token].sum())
 #except Exception:
   #token = '##'+token
   #frams.append(mtx[token])#/mtx[token].sum())
   #ha='##'
    
 j=1
 t = ''+token
 token = t+'.'+str(j)
 while(sel):
  index = sel_index_by_token(t,bert_tokens,j)
  if index != None:
   frams.append(mtx.iloc[index])
   j=j+1
   token = t+'.'+str(j)
  else:
   sel=False
 
 return frams,j,ha

def sel_index_by_token(token,bert_tokens,j=0):
 for i in range(len(bert_tokens)):
  if bert_tokens[i] == token:
   if j >0:
    j=j-1
    #print('j rimossa ', str(j))
   else: 
    return i
  
def update_token(j,has,token):
 if j == 0:
  return str(has) + token
 else: 
  return str(has) + token +'.'+str(j)
 
def comp_token(tokenizer,sentence):
 e=tokenizer.encode(sentence, add_special_tokens=True)
 return tokenizer.convert_ids_to_tokens(e)

def tokens_to_sentence(l1,l2,index=False,dic = None,old_sen=None):
 
 if dic == None:  
   dic = dict()
 
 if len(l2)==0 or len(l1)==0:
  dic['[SEP]']='[SEP]'
  return dic
 
 s = l1.pop(0)
 
 if(s == '[CLS]'):
  dic[s] = '[CLS]'
  tokens_to_sentence(l1,l2,index,dic,old_sen)
  return dic
   
 if(s == '[SEP]'):
  dic[s] = '[SEP]'
  tokens_to_sentence(l1,l2,index,dic,old_sen) 
  return dic 
 
 
 if '##' in  s:
  if index:
   i = 0
   o = s+'_'+str(i)
   while o in dic.keys():
    i=i+1
    o = s+'_'+str(i)
   s = o
    
  dic[s] = old_sen
  l2.pop(0)
  tokens_to_sentence(l1,l2,index,dic,old_sen)
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
  if index:
   i = 0
   if token not in '[SEP]' or token not in '[CLS]':
    o = token+'_'+str(i)
   else:
    o = token 
   while o in dic.keys():
    i=i+1
    o = token+'_'+str(i) 
   token = o
   
  dic[token] = v 
   
 old_sen = v
   
 tokens_to_sentence(l1,l2,index,dic,old_sen)
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
    
def get_all_att_sentece(out_dir,sent_id,token,bert_tokens,id_token=0):
 l = dict()
 for i in range(12):
  for j in range(12):
   frams,k,has = select_sub_matrix_for_token(out_dir,sent_id,str(i+1),str(j+1),token,bert_tokens)
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
   s = select_sub_matrix_for_token(out_dir,name,layer,head,tk,tokens).sum(axis=0)
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
 #att_token = (att_token/att_token.max())*100 
 token_attention = list()
 for i in range(len(bert_tokens)):
  token_attention.append((bert_tokens[i],att_token[i]))
 att = pd.DataFrame(data=token_attention,columns=['token','attention']) 
 return att.sort_values(by='attention',ascending=False)
 
 
 
def get_smear(fram,pos,bert_tokens):
 att_tokens = fram['attention'][pos]
 
 most_important_token = list(fram.loc[fram['attention'] > 5].index)
 print(most_important_token)
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
 token_diff = list()
 div_diff = list()
 # we split l_diff [divergence,token] into [divergence] and [token] list
 for l in l_diff:
  token_diff.append(l[1])
  div_diff.append(l[0]) 
   
 return pd.DataFrame(data=div_diff,index=token_diff,columns=['att_diff']).sort_values(by='att_diff',ascending=False)
  
def comp_jsd(a,n):
 a_prob = np.array(a)
 #print(a_prob)
 b_prob = np.array([1/n]*n)
 #print(b_prob)
 return jensenshannon(a_prob,b_prob)  

def comp_cls(a,n):
 a_prob = np.array(a)
 b_prob = np.zeros(n)
 b_prob[0] = 1
 return jensenshannon(a_prob,b_prob)


def comp_noop(a,n):
 a_prob = np.array(a)
 b_prob = np.zeros(n)
 b_prob[-1] = 1
 return jensenshannon(a_prob,b_prob)

def comp_me(a,n,ind_token,bert_tokens):
 a_prob = np.array(a)
 b_prob = np.zeros(n)
 inds = list()
 
 for i in range(len(bert_tokens)):
  if bert_tokens[i] == bert_tokens[ind_token]:
   inds.append(i)
 
 w = 1/len(inds)
 for i in inds:
  b_prob[i] = w
  
 return jensenshannon(a_prob,b_prob)

def comp_point(a,n,point_ind_list):
 a_prob = np.array(a)
 b_prob = np.zeros(n)
 w = 1/len(point_ind_list)
 for ind in point_ind_list: 
  b_prob[ind] = w
 return jensenshannon(a_prob,b_prob)
 
def comp_score(model):
 dic_score = dict() 
 for j in range(12):
  lay=model.layers[0].encoder.layer[j]
  W0 = lay.attention.dense_output.dense.kernel.numpy()
  nor = list()
  for i in range(12):
   w = W0[i*64:i*64+64,:]
   dic_score[str((j+1,i+1))] =float(np.linalg.norm(w))

 return dic_score


