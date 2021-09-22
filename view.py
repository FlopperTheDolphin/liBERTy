import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from vs_constants import *
from colorama import Fore, Back, Style
from rich.console import Console


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
 
def view_sentence_perc(fram,tokens,token):
  
 console = Console()
 #keys = list(fram_dic.keys())
 dic_perc = dict()
 
 #for token in keys :
 #f = fram.tolist()  
 #network = pd.DataFrame(data=f,columns=[token])
 network = fram
 max_val = network.sort_values(ascending=False).iloc[0]
  
   
  #visualize for each token in keys 
 
 for i in range(len(tokens)):
  if tokens[i] == token:
   x =255 - (network.iloc[i]*255)/max_val
   console.print("[underline]"+token,style="rgb(255,"+str(int(x))+",255)",end=' ', highlight=False)
  else:
   x =255 - (network.iloc[i]*255)/max_val
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
 
 
def view_mtx_pos(dic_edge,dic_pos):
 G = nx.Graph()
 
 for pos in dic_pos.keys():
  G.add_node(pos)
 
 keys = list(dic_pos.keys())
  
 for i in range(len(keys)):
  pos1 = keys[i]
  for j in range(i+1):
   pos2 = keys[j]
   if dic_edge[dic_pos[pos1]+dic_pos[pos2]] > 15:  
    G.add_edge(pos1,pos2,color='r',d=str(pos1)+' '+str(pos2),weight=dic_edge[dic_pos[pos1]+dic_pos[pos2]])
 #  elif dic_edge[dic_pos[pos1]+dic_pos[pos2]] > 10:
  #  G.add_edge(pos1,pos2,color='b',d=str(pos1)+' '+str(pos2),weight=dic_edge[dic_pos[pos1]+dic_pos[pos2]]) 
  # else:
  #  G.add_edge(pos1,pos2,color='g',d=str(pos1)+' '+str(pos2),weight=dic_edge[dic_pos[pos1]+dic_pos[pos2]])         
 
 pos= nx.spring_layout(G)
 
 
 colors = nx.get_edge_attributes(G,'color').values()
 weights = nx.get_edge_attributes(G,'weight').values() 
 
 nx.draw(G,pos,with_labels=True,
 #         width=list(weights),
          edge_color=colors)
 plt.show()	
 
