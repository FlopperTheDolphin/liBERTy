import networkx as nx
import pandas as pd

def view_token(fram,tokens,token):
 print(len(tokens))
 print(fram.size)
 fram = fram.tolist()
 
 network = pd.DataFrame(data=fram,index=tokens,columns=[token])
 
 res = network.sort_values(by=token, ascending=False)
 
 res=res.iloc[:20,:]
 
 #print(network)
 print("---------------20 Parole con più alto valore associato a "+ token +"-----------")
 print(res)
 
 
 
  
 	
 
 
 
