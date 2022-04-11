import os
import math
import numpy as np
from fun.vs_constants import *
from fun.loader import load_from_json,save_in_json
from features.utiliy import get_sentence,get_bert_and_spacy_tokens,get_bert_tokens,find
from fun.comp_att import tokens_to_sentence,select_sub_matrix_for_token,get_head_matrix,get_ordered_token,get_smear,update_token,get_index_from_token,comp_jsd,comp_noop
from fun.view import console_show,view_attention_gradient,view_top_tokens,view_higher_token,view_smear,view_noop,view_matrix,view_total_noop,view_plot
import pandas as pd

def see_attention_sentence(name,layer,head,word,out_dir,model_dir,time=None):
  from features.utiliy import get_sentence,get_bert_and_spacy_tokens
  sentence = get_sentence(out_dir,name)  
  mtx_dir = create_token_path_file(out_dir,name)
  bert_tokens,spacy_tokens = get_bert_and_spacy_tokens(model_dir,sentence,mtx_dir) 
  #dic_sent = unify_bert_and_spacy_tokens(spacy_tokens,bert_tokens)
  #print('dic_sent ',dic_sent)
  #for token in bert_tokens:
   #if dic_sent[token] == word:
  frams,j,has = select_sub_matrix_for_token(out_dir,name,layer,head,word,bert_tokens) 
  if len(frams)==0:
   console_show('no token here, retry')
   print(bert_tokens)
   return
   
  inds = get_index_from_token(word,j,has,bert_tokens,frams)  
  for p in range(len(inds)):
   attentions = inds[p][0]
   #(token,attention,index_in_the_sentence)  
   token_att = list(zip(bert_tokens,attentions,list(range(len(bert_tokens)))))
   max_index = sorted(token_att, key=lambda tup: tup[1], reverse=True)[0][2]
     
   f = 0
   for tup in token_att:
    if word == tup[0]:
      if f == p:
       tk_index = tup[2]
       break
      else:
       f = f+1
     
   console_show('max attention index value:',max_index)
   console_show('index of chosen token:',tk_index)
   console_show('distance in index:',tk_index-max_index)    
   from fun.comp_att import comp_jsd
   console_show('ENTROPY: ' + str(comp_jsd(attentions,len(bert_tokens))))
   console_show('NOOP: ' + str(1-comp_noop(attentions,len(bert_tokens))))  
   plot_gradient_attention(mtx_dir,attentions,bert_tokens,word,layer,head,j) 
   if time != None: 
    view_top_tokens(attentions,bert_tokens,word,time) 
    view_higher_token(attentions,bert_tokens,word,time)  


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
 
def plot_gradient_attention(mtx_dir,fram,bert_tokens,token,layer,head,id_token):
  max_path= os.path.join(mtx_dir,"max.json")
  mx= load_from_json(max_path)
  view_attention_gradient(fram,bert_tokens,token,layer,head,mx,id_token)      

def smear(name,out_dir,model_dir,token,layer,head,id_token,perc):
 sentence = get_sentence(out_dir,name)
 bert_tokens = get_bert_tokens(os.path.join(out_dir, name),model_dir,sentence)
 frams,j,has =select_sub_matrix_for_token(out_dir,name,layer,head,token,bert_tokens)
 
 fram = get_ordered_token(frams[int(id_token)].tolist(),bert_tokens)
 
 n = len(bert_tokens)
 
 jsd = comp_jsd(fram['attention'].to_list(),n)
 print(jsd)
 print('SUM: ' + str(fram['attention'].sum()))
 Res=dict()
 for s in range(n,0,-1):
  #actatt = token not to remove
  actatt = fram[:s]['attention'].tolist()
############################################  
  to = fram[s+1:]['attention'].tolist()
  if len(to) != 0:
   v=0
   for el in to:
    v=v+el
   we = v / len(to)
  else:
   we = 0   
  for i in range(s,n):
   actatt.append(we)
############################################   
  jsd2 = comp_jsd(actatt,n)
  res=jsd2-jsd
  
  Res[s] = res
 
 #print('Results: ' + str(Res)) 
 t = ((1-jsd)/100)*float(perc)
 print('tol: ' + str(t))
 
 pd.set_option('display.max_rows', None)
 print('Take all possible approximations')
 app = None
 for s in Res.keys():
  if abs(Res[s]) <=  t:
   #print(s)
   app = s
 
 print('s: '+str(app) +' word: ' +str(fram.iloc[app]))
 print(fram)
 #print(fram.iloc[0:s]['attention'])
 view_smear(list(Res.keys()),list(Res.values()),len(bert_tokens),app,Res[app])


def extr2(name,out_dir,model_dir,token,layer,head,id_token,perc):
 sentence = get_sentence(out_dir,name)
 bert_tokens = get_bert_tokens(os.path.join(out_dir, name),model_dir,sentence)
 frams,j,has =select_sub_matrix_for_token(out_dir,name,layer,head,token,bert_tokens)
 fram = frams[int(id_token)].tolist() 
 Res = dict()
 c = get_ordered_token(frams[int(id_token)].tolist(),bert_tokens)['attention'].tolist()
 first_jsd = comp_jsd(c,len(bert_tokens))
 tol=((1-first_jsd)/100)*float(perc)
 print('tol: ' + str(tol))
 x=list()
 y=list()
 find_it=False
 r=(len(c),0)
 for s in range(0,len(c)):
  a = list()
  w=0
  o=0
  for j in range(s,len(c)):
   w = w + c[j]
   o=o+1
  w= w/(s+1)
  for j in range(len(c)):
   if s >= j:
    a.append(c[j]+w)  
   else:
    a.append(0)
     
   
  jsd = comp_jsd(a,len(c))
  if abs(jsd-first_jsd) < tol and find_it==False:
    print(jsd)
    print(first_jsd)
    find_it=True
    r=(s,abs(jsd-first_jsd))
    print('s' + str(s))
  x.append(s)
  y.append(abs(jsd-first_jsd))  
 view_plot(x,y,r[0],r[1])   
      
def extr(name,out_dir,model_dir,token,layer,head,id_token,perc):
 
 #sentence = get_sentence(out_dir,name)
 #bert_tokens = get_bert_tokens(os.path.join(out_dir, name),model_dir,sentence)
 #frams,j,has =select_sub_matrix_for_token(out_dir,name,layer,head,token,bert_tokens)
 
 #fram = get_ordered_token(frams[int(id_token)].tolist(),bert_tokens)['attention']
 #l = fram.tolist()
 #n = len(bert_tokens)
 #Res=dict()
 
 #sum = 0
 #for s in range(len(l)):
 # sum=sum+l[s]
 # if sum > float(perc):
 #  break
   
 #print('Valori importanti: ' + str(s+1))
 #print(fram.iloc[:s+1])
   
 sentence = get_sentence(out_dir,name)
 bert_tokens = get_bert_tokens(os.path.join(out_dir, name),model_dir,sentence)
 frams,j,has =select_sub_matrix_for_token(out_dir,name,layer,head,token,bert_tokens)
 
 fram = get_ordered_token(frams[int(id_token)].tolist(),bert_tokens)
 first_jsd = comp_jsd(fram['attention'].tolist(),len(bert_tokens)) 
 n = len(bert_tokens)
 Res=dict()
 not_find=False
 tol=((1-first_jsd)/100)*float(perc)
 print('tol: ' + str(tol))
 for s in range(0,n-1):
  new_df = fram.drop(list(range(0,s+1)))
  g=list()
  for i in range(s+1):
   g.append(fram.iloc[i]['attention'])
  sm=sum(g)
  att_remained = len(new_df['attention'].tolist())
  att_to_add = sm/len(new_df['attention'].tolist())
  print('attention removed: ' + str(len(g)))
  print('attentions left: ' + str(att_remained))
  print('total probability to replace: ' + str(sm))
  print('att to add: ' + str(att_to_add))
  
  actatt = new_df['attention'].tolist()
  for att in actatt:
   att= att+att_to_add
   
  j=comp_jsd(actatt,len(actatt))
  q = abs(j - first_jsd)
  print('diff: ' + str(q))
  Res[s] = q
  
  if q > tol and not_find==False:
   not_find=True
   print(str(q) + ' yesssssssss')
   chosen = s
 view_plot(Res.keys(),Res.values(),chosen,Res[chosen])  
  
