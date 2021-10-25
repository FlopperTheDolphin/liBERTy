import os
from fun.vs_constants import *
from features.utiliy import get_sentence,get_bert_tokens,find
from fun.loader import load_from_json,save_in_json
from fun.view import console_show,view_token_div,view_chosen_heads,view_chosen_tokens,view_total_stat, view_cartesian_div,view_outlier,view_total_outlier
from fun.comp_att import get_all_att_sentece,comp_divergence,comp_avg_and_std,update_matrix,get_head_matrix,get_max,sort_outliers


def see_stat(name,out_dir,token,model_dir,vector,id_token):
  sentence,bert_tokens = get_sentence_and_bert_tokens(out_dir,name,model_dir)
  dic_att=from_token_select_all_columns(out_dir,name,token,id_token)
  n_token = len(bert_tokens)
  if vector == None or vector== 'entropy':
   df,l,A=comp_divergence(dic_att,n_token,bert_tokens)
   view_token_div(df,token,l,A,id_token)
  elif vector == 'noop':
   weight = ['[CLS]','[SEP]','.']
   df,l,A=comp_divergence(dic_att,n_token,bert_tokens,weight)
   view_token_div(df,token,l,A,id_token)
  elif vector == 'all':
  #for now we have only two vectros so we use cartesian plot
   console_show(MSG_COMPARE)
   weight = ['[CLS]','[SEP]','.']
   df1,l1,A1=comp_divergence(dic_att,n_token,bert_tokens,weight)
   console_show(MSG_DONE_1)  
   df2,l2,A2=comp_divergence(dic_att,n_token,bert_tokens)
   console_show(MSG_DONE_2)
   view_cartesian_div(df1,df2,l1,id_token,token)
   

def see_noop(name,out_dir,token,model_dir):
 sentence,bert_tokens = get_sentence_and_bert_tokens(out_dir,name,model_dir)
 dic_att=from_token_select_all_columns(out_dir,name,token)
 n_token = len(bert_tokens)
 df,l,A=comp_divergence(dic_att,n_token,vec='find_cls')
 view_token_div(df,token,l,A)

def comp_stat(name,out_dir,perc,model_dir,vector='entropy'):

  sentence,bert_tokens = get_sentence_and_bert_tokens(out_dir,name,model_dir)
  mn,mx,freq_path,mat_token_path=initialise_perc_and_define_paths(out_dir,name,perc)
  
  if not os.path.exists(freq_path) or not os.path.exists(mat_token_path):
    un_tokens = list(set(bert_tokens))
    dic_head,dic_token,n_token=get_dictionaries_and_length(bert_tokens)
    for token in un_tokens:
     console_show(TOKEN,token)
     possible_id = find(sentence,bert_tokens,token,view=False)
     for id_token in possible_id:
      print(ID_TOKEN,id_token)
      heads_chosen,df = select_head_by_mn_mx(out_dir,name,token,n_token,mn,mx,id_token,bert_tokens,vector)
      dic_head,dic_token=create_file_freq_and_save(heads_chosen,dic_head,dic_token,token,freq_path,mat_token_path)
  else:
    dic_token = load_from_json(mat_token_path)
    dic_head = load_from_json(freq_path)
    
  view_chosen_heads(dic_head,dic_token,mn,mx,bert_tokens) 
    
  # Parte aggiunta per controllare se per caso in ogni token esistesse qualche informazione tra le head trovate
  # infatti sono state prese in considerazione le head con una frequenza minore nel calcolo max di divergenze di shannon
  # purtroppo niente di nuovo, anzi le head hanno mostrato fenomeni di offset singolari in quelle head isolate

def initialise_total_path(out_dir,name,vector):
 return os.path.join(os.path.join(out_dir,name),"total_freq_"+str(vector)+".json")
 
def total_stat(name,out_dir,model_dir,vector,view=True):
 sentence,bert_tokens = get_sentence_and_bert_tokens(out_dir,name,model_dir)
 total_path = initialise_total_path(out_dir,name,vector)
 
 if check_file_exists(total_path) == False:
  console_show(SEPARATOR,pick=False)
  console_show(MSG_TOTAL_STAT)
  console_show(MSG_WARNING_TIME)
  console_show(SEPARATOR,pick=False)
  
  dic_total = dict()
  u_tokens = list(set(bert_tokens))
  for token in u_tokens:
   console_show(TOKEN,token)
   
   possible_id = find(sentence,bert_tokens,token,view=False)
   
   for id_token in possible_id:
    console_show(ID_TOKEN,id_token)
    dic_att = get_all_att_sentece(out_dir,name,token,id_token)
    if vector == None or vector == 'entropy':
     df,index_list,A=comp_divergence(dic_att,len(bert_tokens),bert_tokens)
    elif vector == 'noop':
     weight = ['[CLS]','[SEP]','.']
     df,index_list,A=comp_divergence(dic_att,len(bert_tokens),bert_tokens,weight)
   
   #print(df)
    for index in index_list:
     if index not in dic_total.keys():
      dic_total[index] = list()  
     dic_total[index].append(df['divergence'][index])
    
   save_in_json(dic_total,total_path)
 else:
  dic_total = load_from_json(total_path)  
  view_total_stat(dic_total)
  #index_list = list()
  #for i in range(12):
   #for j in range(12):
    #index_list.append(str((i+1,j+1)))
  #A = get_head_matrix()  
 
 #B = A.copy()
  
 #for index in dic_total.keys():
 # div = dic_total[index]
 # avg,std=comp_avg_and_std(div)
  
#  dic_total[index] = (avg,std)
  
 #we recicle matrix A to another use here 
 
 #for i in range(12):
 # for j in range(12):
 #  for index in index_list:
  #  if str((i+1,j+1)) in index:
  #   A = update_matrix(A,i,j,dic_total[index][0])
  #   B = update_matrix(B,i,j,dic_total[index][1])
 
 #max_a = get_max(A)
 #max_b = get_max(B)
     
 #if view == True:    
  
 
 #return dic_total,index_list

def get_sentence_and_bert_tokens(out_dir,name,model_dir):
  sentence = get_sentence(out_dir,name)
  mtx_dir = create_token_path_file(out_dir,name)
  bert_tokens=get_bert_tokens(mtx_dir,model_dir,sentence)
  return sentence,bert_tokens

def select_head_by_mn_mx(out_dir,name,token,n_token,mn,mx,id_token,bert_tokens,vector='entropy'):
  dic_att = get_all_att_sentece(out_dir,name,token,id_token)

  if vector == None or vector == 'entropy':
     df,index_list,A=comp_divergence(dic_att,len(bert_tokens),bert_tokens)
  elif vector == 'noop':
     weight = ['[CLS]','[SEP]','.']
     df,index_list,A=comp_divergence(dic_att,len(bert_tokens),bert_tokens,weight)
     
  heads_chosen = list(df.loc[(df['divergence'] >= (int(mn)/100)) & (df['divergence'] <= (int(mx)/100))].index.values)
  console_show(MSG_HEAD_CHOSEN,heads_chosen)
  return heads_chosen,df
        
def create_file_freq_and_save(heads_chosen,dic_head,dic_token,token,freq_path,token_path):  
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

 
def from_token_select_all_columns(out_dir,name,token,id_token=0):
 console_show(MSG_SENT,name)
 console_show(MSG_COMP_COLUMN)
 dic_att= get_all_att_sentece(out_dir,name,token,id_token) 
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


def outlier(name,layer,head,out_dir,model_dir,vector,view=True,sentence=None,bert_tokens=None):
 if sentence==None or bert_tokens == None:
  sentence,bert_tokens = get_sentence_and_bert_tokens(out_dir,name,model_dir)
 total_path = initialise_total_path(out_dir,name,vector)
 if check_file_exists(total_path) == False:
  console_show(MSG_EXEC_STAT,vector)
 else:
  dic_total = load_from_json(total_path)
  jsd = dic_total['('+str(layer)+', '+str(head)+')']
  avg = sum(jsd)/len(jsd)
  l_diff = list()
  for i in range(len(jsd)): 
   diff_token = abs(jsd[i] - avg)
   l_diff.append(diff_token)
  df = sort_outliers(l_diff,bert_tokens)
  if view == True:     
   view_outlier(df)
 tokens =''  
 for i in range(10):  
  tokens = tokens + ' ' + str(df.iloc[i].name)
 #print(df.iloc[2].name)   
 return (tokens,df['att_diff'].iloc[:10].sum()/10)#df.iloc[:5].sum()/5 

#def check_avg_path():

def total_outlier(name,out_dir,model_dir,vector):
 sentence,bert_tokens = get_sentence_and_bert_tokens(out_dir,name,model_dir)
 dic_diff = dict()
 index_list=list()
 for i in range(1,13,1):
   for j in range(1,13,1):
    index = '('+str(i)+', '+str(j)+')'
    dic_diff[index] = outlier(name,i,j,out_dir,model_dir,vector,False,sentence,bert_tokens)
    index_list.append(index)
 view_total_outlier(dic_diff,bert_tokens,index_list)   
      
