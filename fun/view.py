import networkx as nx
from networkx.readwrite import json_graph
import pandas as pd
import matplotlib.pyplot as plt
from fun.vs_constants import *
from rich.console import Console


def view_top_tokens(fram,tokens,token,time):
 console = Console()
 print('') 
  
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
   
   console.print(CHOSEN_WORD+tk,style=color,end=' ', highlight=False)
      
  elif tk in green:
    console.print(tk,style=GREEN,end=' ', highlight=False)
  elif tk in yellow :
    console.print(tk,style=YELLOW,end=' ', highlight=False)
  elif tk in red:
    console.print(tk,style=RED,end=' ',  highlight=False)
  else:
    console.print(tk,end=' ',highlight=False)  
  

 space()   
 
 console.print(SEPARATOR)
 console.print(CHOSEN_WORD+UNDERLINE,end=' ', highlight=False) 
 console.print(MSG_AN_TOKEN)
 console.print(MSG_RED ,style=RED,end=' ', highlight=False) 
 console.print(msg_red(t,token),highlight=False)
 console.print(MSG_YELLOW,style=YELLOW,end=' ', highlight=False) 
 console.print(msg_yellow(t,token),highlight=False)
 console.print(MSG_GREEN,style=GREEN,end=' ', highlight=False) 
 console.print(msg_green(t,token),highlight=False)
 console.print(SEPARATOR)
 
 space()
 space()
 
def view_attention_gradient(fram,tokens,token,layer,head,mx):

 console = Console()
 dic_perc = dict()
 network = fram
 max_val = network.sort_values(ascending=False).iloc[0]
  
   
  #visualize for each token in keys 
 m = mx[str((int(layer),int(head)))]
 console_show(MSG_MAX_MTX,m)
 space()
 
 for i in range(len(tokens)):
  if tokens[i] == token:
   x =255 - (network.iloc[i]*255)/max_val
   console.print("[underline]"+token,style="rgb(255,"+str(int(x))+",255)",end=' ', highlight=False)
  else:
   x =255 - (network.iloc[i]*255)/max_val
   #(255,0/255,0)
   console.print(str(tokens[i]),style="rgb(255,"+str(int(x))+",255)",end=' ', highlight=False)
      
 space()
 space()
 

def view_higher_token(fram,tokens,token,time=None):

 console = Console()
 if time == None:
  t = 20
 else:
  t = time 
  
 f = fram.tolist()
 network = pd.DataFrame(data=f,index=tokens,columns=[token])
 res = network.sort_values(by=token, ascending=False)
 res=res.iloc[:int(t),:]
 console.print(higher_token(t,token),highlight=False)
 console.print(res,highlight=False)
 space()
 
 
def view_mtx_pos(dic_edge,dic_pos,path_cache,path_img,save):
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
 
 for pos in dic_pos.keys():
  w = (dic_edge[dic_pos[pos]+dic_pos[pos]])*100/max_v
  s=0
  for pos2 in dic_pos.keys(): 
   s = s+ dic_edge[dic_pos[pos]+dic_pos[pos]]*25
  if w < 50: 
   G.add_node(pos,color_node='green',size=int(s))
  elif w < 75:
   G.add_node(pos,color_node='yellow',size=int(s))
  else:
   G.add_node(pos,color_node='red',size=int(s))  


 for i in range(len(keys)):
  pos1 = keys[i]
  for j in range(i+1):
   pos2 = keys[j]
   
   w = (dic_edge[dic_pos[pos1]+dic_pos[pos2]])*100/max_v  
   if w >= 75 and w < 90:
    if pos1 != pos2:
     G.add_edge(pos1,pos2,label='',color='blue')  
    else:
     G.add_edge(pos1,pos2,label='',color='blue')   
   elif w >=90: 
    if pos1 != pos2:
     G.add_edge(pos1,pos2,label='',color="red")  
    else:
     G.add_edge(pos1,pos2,label='',color="red")   
 
 draw_graph(G,path_img,save)
 return json_graph.node_link_data(G)	
 
def view_loaded_pos(path,path_img,save):
  try:  
   data = load_from_json(path)
   G=json_graph.node_link_graph(data) 
   draw_graph(G,path_img,save)
   return True
  except Exception:
   return False 


def draw_graph(G,path_img,save):

 pos = nx.circular_layout(G)

 colors = nx.get_edge_attributes(G,'color').values()
 weights = nx.get_edge_attributes(G,'weight').values() 
 labels =  nx.get_edge_attributes(G,'label')
 color_node = nx.get_node_attributes(G,'color_node').values()
 node_size = list(nx.get_node_attributes(G,'size').values())
 
 nx.draw(G,pos=pos,with_labels=True,#,
           node_color=color_node,
           edge_color=colors,
           node_size=node_size)
    
 nx.draw_networkx_edge_labels(G, pos=pos, edge_labels = labels)          
           
 if save == True:
    plt.savefig(path_img)
    console_show(MSG_GRAPH_SAVED,path_img)

 else:
  plt.show()  	
 
 plt.close()
  
def view_token_div(df,token,list_index,A):
 console_show(TOKEN,token)
 space()
 
 console = Console()
 df=df.sort_values(by='divergence',ascending=False) 
 max_v=df.iloc[0,0]
 j=1
 
 for i in range(len(list_index)): 
  x =255 - (df.loc[list_index[i]]*255)/max_v
   #(255,0/255,0)
  console.print('$',style="rgb(255,"+str(int(x))+",255)",end=' ', highlight=False)  
  if((i+1)%12 == 0):
   space()
   
 space()
 console.print(SEPARATOR)  
 console_show(MSG_FIRST_DIV_MTX,token,False) 
 console.print(df.iloc[0:10,:],highlight=False)
 console.print(SEPARATOR) 

 #print(A)
 #print(df)
 view_matrix(A,True)
 #plt.matshow(A)
 #plt.show() 
#def view_att_total(name,list_weight,tokens):
# console=Console()
# res=pd.DataFrame(data=list_weight,index=tokens,columns=['tokens']).sort_values(by='tokens',ascending=False)
# max_v = res.iloc[0,0]
# for i in range(len(tokens)): 
#  x =255 - (list_weight[i]*255)/max_v
   #(255,0/255,0)
#  console.print(str(tokens[i]),style="rgb(255,"+str(int(x))+",255)",end=' ', highlight=False)
  
# print('')
# print('----------------------------------')    
# print('Primi 10 token:')
# print(res.iloc[0:10,:])
# print('----------------------------------') 

   
#standardize all msg in terminal    
def console_show(msg,ob=None,pick=True):
 if ob == None:
  if pick == True:
   print('> ' + msg)
  else:
   print(msg)
 else:
  if pick == True: 
   print('> ' + msg + ' ' + str(ob)) 
  else:
   print(msg + ' ' + str(ob)) 

def space():
 print('')  
 

def view_chosen_heads(dic_head,dic_token,mn,mx,tokens):   
 heads_sort = sorted(dic_head.items(), key=lambda x: x[1])
 heads_sus = heads_sort[0:3]   
 console_show(MSG_HEAD_SORTED,heads_sort)
   
 for token in tokens:
  heads_list = dic_token[token]
  for head in heads_list:
   for i in range(len(heads_sus)):
    if head in heads_sus[i]:
     console_show(MSG_TOKEN_SUS,token,False)
     console_show(MSG_HEADS_SUS,head,False)
     console_show(SEPARATOR,None,False)
        
def view_chosen_tokens(layer,head,tokens,dic_token):
 console=Console()        
 head_n = '('+str(layer)+', '+str(head)+')'
 console_show(MSG_CHOSEN_TOKEN_GIVEN_HEAD,head_n)
 for token in tokens:
  if head_n in dic_token[token]: 
   console.print(str(token),end=' ',highlight=False)
 space()  
 
def view_dist(dist,l_index):
 l_sort = sorted(dist.items(), key=lambda x: x[1])
 max_v = l_sort[-1][1]
 
 for element in l_sort:
  console_show(element[0]+':',(element[1]/max_v),pick=False)
  
  
def view_matrix(A,col=False):
 figure = plt.figure()
 axes = figure.add_subplot(111)
 caxes = axes.matshow(A)
 if col == True:
  figure.colorbar(caxes)
 plt.show()

#def view_cluster(count,ind,l_index,bins):
# ind = sorted(ind.items(), key=lambda x: x[1])
 
# print(ind)
 
# print()
# print(count)
# plt.hist(count)
# plt.show()

def view_interp(x_tokens,x_grid,y_token,y_grid):
 plt.plot(x_tokens,y_token,'o',x_grid,y_grid,'-')
 plt.show()
  
