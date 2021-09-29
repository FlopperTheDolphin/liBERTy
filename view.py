import networkx as nx
from networkx.readwrite import json_graph
import pandas as pd
import matplotlib.pyplot as plt
from vs_constants import *
#from colorama import Fore, Back, Style
from rich.console import Console
from loader import save_in_json, load_from_json
def view_sentence_time(fram,tokens,token,time):
 console = Console()
 print('') 
# keys = list(fram_dic.keys())
  
 red = list()
 yellow=list()
 green=list()
 
 t=int(time)
  
    
 network = pd.DataFrame(data=fram.tolist(),index=tokens,columns=[token])  
 res = network.sort_values(by=token,ascending=False)
 res_green = res.iloc[t+t+1:t+t+t+1]
 res_yellow = res.iloc[t+1:t+t+1]
 res_red = res.iloc[:t]
 green=green+list(res_green.index)
 yellow=yellow+list(res_yellow.index)
 red=red+list(res_red.index)
  

 for tk in tokens:
  
  if tk == token :
  
   color = ''
   if tk in green:
     color = GREEN  
   elif tk in yellow :
     color = YELLOW  
   elif tk in red:
     color = RED     
   
   #print(Style.RESET_ALL,end='')
   console.print(CHOSEN_WORD+tk,style=color,end=' ', highlight=False)
   #print(Style.RESET_ALL,end='')
      
  elif tk in green:
    console.print(tk,style=GREEN,end=' ', highlight=False)
  elif tk in yellow :
    console.print(tk,style=YELLOW,end=' ', highlight=False)
  elif tk in red:
    console.print(tk,style=RED,end=' ',  highlight=False)
  else:
    print(tk,end=' ')  
  
# print(Style.RESET_ALL)
 print('')    
 
 print('--------------------------------------------------------------------')
 console.print(CHOSEN_WORD+'SOTTOLINEATO',end=' ', highlight=False) 
 print('= token analizzato')
 console.print('ROSSO' ,style=RED,end=' ', highlight=False) 
 print('= i primi '+ str(t)+' token con il più alto valore di attention per ' + token)
 console.print('GIALLO',style=YELLOW,end=' ', highlight=False) 
 print('= da '+ str(int(t+1)) +' a '+str(int(t+t))+' token con più al valore di attention per ' + token)
 console.print('VERDE',style=GREEN,end=' ', highlight=False) 
 print('= da '+ str(int(t+t+1)) +' a '+str(int(t+t+t))+' token con il più alto valore attention di per ' + token)
 print('--------------------------------------------------------------------')
 
 print('')
 print('')
 
def view_sentence_perc(fram,tokens,token,layer,head,mx):
  
 console = Console()
 #keys = list(fram_dic.keys())
 dic_perc = dict()
 
 #for token in keys :
 #f = fram.tolist()  
 #network = pd.DataFrame(data=f,columns=[token])
 network = fram
 max_val = network.sort_values(ascending=False).iloc[0]
  
   
  #visualize for each token in keys 
 m = mx[str((int(layer),int(head)))]
 print('Valore di attention massimo nella matrice: ' + str(m))
 print('')
 for i in range(len(tokens)):
  if tokens[i] == token:
   x =255 - (network.iloc[i]*255)/max_val
   #x =255 - (network.iloc[i]*255)/m
   console.print("[underline]"+token,style="rgb(255,"+str(int(x))+",255)",end=' ', highlight=False)
  else:
   x =255 - (network.iloc[i]*255)/max_val
   #x =255 - (network.iloc[i]*255)/m
   #(255,0/255,0)
   console.print(str(tokens[i]),style="rgb(255,"+str(int(x))+",255)",end=' ', highlight=False)
      
 print('')
 print('')
 

def view_word(fram,tokens,token,time=None):

 if time == None:
  t = 20
 else:
  t = time 
  
 
 f = fram.tolist()
 network = pd.DataFrame(data=f,index=tokens,columns=[token])
 res = network.sort_values(by=token, ascending=False)
 res=res.iloc[:int(t),:]
 print('Primi ' +str(t)+' token con il più alto valore di attention per ' + token)
 print(res)
 print('')
 
 
def view_mtx_pos(dic_edge,dic_pos,path_cache,path_graph):
 G = nx.Graph()
 
 keys = list(dic_pos.keys())
 
 l=list()
 for i in range(len(keys)):
  pos1 = keys[i]
  for j in range(i+1):
   pos2 = keys[j]
   l.append(dic_edge[dic_pos[pos1]+dic_pos[pos2]])
 
 
 l.sort(reverse=True)
 max_v = l[0]
 print(str(max_v))   
 
 for pos in dic_pos.keys():
  w = (dic_edge[dic_pos[pos]+dic_pos[pos]])*100/max_v  
  if w < 50: 
   G.add_node(pos,color_node='green')
  elif w < 75:
   G.add_node(pos,color_node='yellow')
  else:
   G.add_node(pos,color_node='red')  
   
  
 for i in range(len(keys)):
  pos1 = keys[i]
  for j in range(i+1):
   pos2 = keys[j]
   
   w = (dic_edge[dic_pos[pos1]+dic_pos[pos2]])*100/max_v  
   #if w < 50:
   # G.add_edge(pos1,pos2,label='',weight=float(w),color="green")
   #if w < 75:
    #if pos1 != pos2:
     #G.add_edge(pos1,pos2,label=str(pos1)+' '+str(pos2),weight=float(w),color="yellow")
    #else:
     #G.add_edge(pos1,pos2,label='',weight=float(w),color="yellow")
   if w >= 75:
    if pos1 != pos2:
     G.add_edge(pos1,pos2,label=str(pos1)+' '+str(pos2),weight=float(w),color="red")  
    else:
     G.add_edge(pos1,pos2,label='',weight=float(w),color="red")   
 

 data = json_graph.node_link_data(G)
 save_in_json(data,path_graph)
 
 draw_graph(G)	
 
def view_loaded_pos(path):
  try:  
   data = load_from_json(path)
   G=json_graph.node_link_graph(data) 
   draw_graph(G)
   return True
  except Exception:
   return False 


def draw_graph(G):
 pos= nx.spring_layout(G)
 
 colors = nx.get_edge_attributes(G,'color').values()
 weights = nx.get_edge_attributes(G,'weight').values() 
 labels =  nx.get_edge_attributes(G,'label')
 color_node = nx.get_node_attributes(G,'color_node').values()

 nx.draw(G,pos,with_labels=True,#,
           node_color=color_node,
           edge_color=colors)
           
 nx.draw_networkx_edge_labels(G, pos, edge_labels = labels)          
           
           
 plt.show()	
  
def view_token_div(df,tk,list_index):
 print('TOKEN: ' +tk)
 print('')
 console = Console()
 df=df.sort_values(by='divergence',ascending=False) 
 max_v=df.iloc[0,0]
 j=1
 for i in range(len(list_index)): 
  x =255 - (df.loc[list_index[i]]*255)/max_v
   #(255,0/255,0)
  console.print('$',style="rgb(255,"+str(int(x))+",255)",end=' ', highlight=False)
  
  if((i+1)%12 == 0):
   print('')
   
 print('')
 print('----------------------------------')    
 print('Prime 10 matrici più divergenti:')
 print(df.iloc[1:10,:])
 print('----------------------------------') 

  
def view_att_total(name,list_weight,tokens):
 console=Console()
 res=pd.DataFrame(data=list_weight,index=tokens,columns=['tokens']).sort_values(by='tokens',ascending=False)
 max_v = res.iloc[0,0]
 for i in range(len(tokens)): 
  x =255 - (list_weight[i]*255)/max_v
   #(255,0/255,0)
  console.print(str(tokens[i]),style="rgb(255,"+str(int(x))+",255)",end=' ', highlight=False)
  
   
 print('')
 print('----------------------------------')    
 print('Primi 10 token:')
 print(res.iloc[1:10,:])
 print('----------------------------------') 

   
    



