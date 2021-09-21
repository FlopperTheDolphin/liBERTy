import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

def view_token(fram,tokens,token):
 fram = fram.tolist()
 
 network = pd.DataFrame(data=fram,index=tokens,columns=[token])
 
 res = network.sort_values(by=token, ascending=False)
 
 res=res.iloc[:20,:]
 
 #print(network)
 print(VW_20_TOKENS + token + VW_20_TOKENS2)
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
 
