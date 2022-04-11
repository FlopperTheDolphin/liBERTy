from features.utiliy import get_sentence,get_bert_tokens,find
import os
from fun.comp_att import select_sub_matrix_for_token,get_index_from_token,comp_noop,comp_jsd,comp_me,get_head_matrix,update_matrix
from fun.view import console_show_color_red,console_show_color_black,console_show_color_green,view_matrix
def good_head(name,layer,head,out_dir,model_dir,bert_tokens=None,sentence=None,verbose=True): 
 if bert_tokens == None or sentence == None:
  sentence = get_sentence(out_dir,name)  
  mtx_dir = os.path.join(out_dir, name)
  bert_tokens = get_bert_tokens(mtx_dir,model_dir,sentence)
 tokens = list(set(bert_tokens))
 good = list()
 wrong=list()
 others=list()
 ents=list()
 
 for token in tokens:
  frams,j,has = select_sub_matrix_for_token(out_dir,name,layer,head,token,bert_tokens) 
  possible_id = find(sentence,bert_tokens,token,view=False)
  for id_token in possible_id:
   noop = comp_noop(frams[id_token],len(bert_tokens))  
   ent = comp_jsd(frams[id_token],len(bert_tokens))
   me = comp_me(frams[id_token],len(bert_tokens),id_token,bert_tokens)
   ents.append(ent)
 
   if noop > 0.7:
    good.append(str(token) + '.' + str(id_token))
   if noop < 0.5:
    wrong.append(str(token) + '.' + str(id_token))
   else:
    others.append(str(token) + '.' + str(id_token)) 
   
     
 if verbose == True:
 ##################################
  markup = list()
  first_row = "\\newline \\newline {\\tiny \\begin{tcolorbox}[colback=white,title=\\textbf{head: ("+str(layer)+"-"+str(head)+") token: "+ str(token)+"},colbacktitle=red] \\texttt {{\\tiny"
  last_row = "}}\end{tcolorbox}}"
  markup.append(first_row)
################################################ 
  d_token = dict()
  for token in bert_tokens:
    try:
     j = d_token[token]
     d_token[token] = d_token[token] + 1
    except Exception:
     j=0
     d_token[token] = 1
    t = str(token) + '.' + str(j)
    
    tok = token.replace('#','\#')
    
    
    if t in good:
      console_show_color_green(token)
     ################################################## 
      markup.append("{\color[RGB]{0,255,0} "+str(tok)+"}")   
     #################################################### 
    elif t in wrong:
      console_show_color_red(token)  
    ######################################################ààà
      markup.append("{\color[RGB]{255,0,0} "+str(tok)+"}")
    #################################################  
    else:
      console_show_color_black(token) 
    ########################################################àààà
      markup.append(str(tok))
    ###################  
  print('avg entropy: ' + str(sum(ents)/len(ents)) )
  
  markup.append(last_row) 
 #######################################################
 
  path=str(layer)+","+str(head)+"_"+str(token)+".tex"
  f = open(path, "a")
  for m in markup:
   f.write(m +" ")
  f.close()
  print("> laTex file created at ["+path+"]")
############################################################### 
 
  
  #print(others)     
 return len(good),len(wrong),len(others)
 
def clas_noop(name,out_dir,model_dir,verbose=True):
 sentence = get_sentence(out_dir,name)  
 mtx_dir = os.path.join(out_dir, name)
 bert_tokens = get_bert_tokens(mtx_dir,model_dir,sentence)
 m = len(bert_tokens)
 A = get_head_matrix()
 pure_index=list()  
 for i in range(12):
  for j in range(12):
   print(str((i+1,j+1)),end=' ')
   good,wrong,others = good_head(name,str(i+1),str(j+1),out_dir,model_dir,bert_tokens,sentence,False) 
   good = good + others
   if good > wrong:
    A = update_matrix(A,i,j,1)
    #print(1)
    pure_index.append(str((i+1,j+1)))
   else: 
    A = update_matrix(A,i,j,-1)
    #print(-1) 
 if verbose == True:   
  view_matrix(A,col=False)
 else:
  return pure_index

