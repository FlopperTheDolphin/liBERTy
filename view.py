import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from vs_constants import *
from colorama import Fore, Back, Style

def view_sentence(fram_dic,tokens,word,spacy_token,time=None):
 print('------------------------------------------------') 
 keys = list(fram_dic.keys())
 
 red = list()
 yellow=list()
 green=list()
 
 if time == None:
  t = 20
 else:
  t = int(time)
  
    
 
 for token in keys :
  f = fram_dic[token].tolist()  
  network = pd.DataFrame(data=f,index=tokens,columns=[token])
  res = network.sort_values(by=token, ascending=False)
  res_green = res.iloc[t+t+1:t+t+t,:]
  res_yellow = res.iloc[t+1:t+t,:]
  res_red = res.iloc[:t,:]
  green=green+list(res_green.index)
  yellow=yellow+list(res_yellow.index)
  red=red+list(res_red.index)
  
  
 for token in tokens:
  if token in keys:
  
   color = ''
   if token in green:
     color = Fore.GREEN  
   elif token in yellow :
     color = Fore.YELLOW  
   elif token in red:
     color = Fore.RED     
   
   print(Style.RESET_ALL,end='')
   print(Back.BLUE + color + token,end=' ')
   print(Style.RESET_ALL,end='')
      
  elif token in green:
     print(Fore.GREEN + token,end=' ')  
  elif token in yellow :
     print(Fore.YELLOW + token,end=' ')
  elif token in red:
     print(Fore.RED + token,end=' ')
  else:
    print(Style.RESET_ALL + token,end=' ')  
  
 print(Style.RESET_ALL)    
 
 print('---------------------------------------------------------------')
 print('* '+Back.BLUE+'BLUE'+Style.RESET_ALL+ ' = token analizzati')
 print('* '+Fore.RED+'ROSSO'+Style.RESET_ALL+' = token con il più alto tasso di attention')
 print('* '+Fore.YELLOW+'GIALLO'+Style.RESET_ALL+' = token con un tasso medio di attention')
 print('* '+Fore.GREEN+'VERDE'+Style.RESET_ALL+' = token con un tasso minore di attention')
 print('---------------------------------------------------------------')
 
 

def view_word(fram_dic,tokens,word,time=None):

 print('token individuati per la parola ' + word)

 if time == None:
  t = 20
 else:
  t = time 
  
 
 for token in list(fram_dic.keys()):
  f = fram_dic[token].tolist()
  network = pd.DataFrame(data=f,index=tokens,columns=[token])
  res = network.sort_values(by=token, ascending=False)
  res=res.iloc[:int(t),:]
 
 #print(network)
  print('--------------- Parole con la più alta corrispondenza ' + token + ' --------------')
  print(res)
 
 
  
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
 
