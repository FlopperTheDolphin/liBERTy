from sklearn.cluster import MeanShift,AgglomerativeClustering
import numpy as np
from features.utiliy import get_sentence,get_bert_tokens
from fun.comp_att import select_sub_matrix_for_token,get_index_from_token
import os
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import scipy

def smear(head,layer,name,token,out_dir,model_dir,verbose=True,id_token=0):
 sentence = get_sentence(out_dir,name)  
 mtx_dir = os.path.join(out_dir, name)
 bert_tokens = get_bert_tokens(mtx_dir,model_dir,sentence) 
 
 #print(out_dir,name,layer,head,token,bert_tokens)
 
 frams,j,has = select_sub_matrix_for_token(out_dir,name,layer,head,token,bert_tokens) 
  
 #print(frams)
 att=np.array(frams[id_token]).reshape(-1,1)
 #print(att) 
  #print(att) 
 
 clustering = MeanShift().fit(np.array(att))
  #print(clustering.labels_)
  
 if verbose==True:
  print(clustering.labels_)
 old_tokens=dict()
 clust = dict()
 for i in range(len(clustering.labels_)):
  if bert_tokens[i] in old_tokens.keys():
    token = bert_tokens[i]+'.'+str(old_tokens[bert_tokens[i]])
    old_tokens[bert_tokens[i]] = old_tokens[bert_tokens[i]] +1 
  else: 
    token = bert_tokens[i]
    old_tokens[bert_tokens[i]] = 1
  try: 
   clust[clustering.labels_[i]].append((token,frams[id_token][i]))
  except Exception:
   clust[clustering.labels_[i]]=list()
   clust[clustering.labels_[i]].append((token,frams[id_token][i]))
  
 means = dict()  
  #print(clust)
 for k in clust.keys():
  #print(k)
  if len(clust[k])!=0:
   atts =list()
   for att in clust[k]: 
    atts.append(att[1])
     
   means[k] = scipy.mean(atts, axis=0)  
   
  
 means = sorted(means.items(), key=lambda x: x[1])
  
 if verbose == True:
  sol = dict()
  for i in range(len(means)):
   sol[i] = clust[means[i][0]]
   print()
   print(i,clust[means[i][0]])
  
  
 x=list()
 y=list()
 old_tokens=dict()
 for j in range(len(bert_tokens)):
  tk = bert_tokens[j]
  if tk in old_tokens.keys():
   token = tk+'.'+str(old_tokens[tk])
   old_tokens[tk]=old_tokens[tk]+1
  else:
   token = tk
   old_tokens[tk]=1
  for i in range(len(means)):
   q = list()
   for cl in clust[means[i][0]]:
    q.append(cl[0])
   if token in q:
     y.append(i)
  x.append(j)   
    
  #plot things  
  #if verbose == True:
   #print(x,y)
  #plt.bar(x,y,width=1)
   #plt.step(x,y)
   #plt.show()
    
 return means,clust 
  #model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

 # model = model.fit(att)
 # plot_dendrogram(model, truncate_mode="level", p=3)
  
 # plt.show()

 #return
