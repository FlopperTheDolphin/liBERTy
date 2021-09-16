#import torch
#import transformers
#import numpy as np
import pandas as pd
#from transformers import AutoModel,AutoTokenizer, BertConfig
#import networkx as nx


#VISUALIZZAZIONE
df = pd.read_csv("pr/layer-1_head-1.csv")

print(df)

#ref = "TC TORACE-ADDOME Esame condotto prima e dopo somministrazione di mdc endovena (Xenetix 350, 120 ml), confrontato con precedente del 12/11/2018. TORACE: rimossi la cannula tracheale ed il drenaggio toracico destro. Non significative modificazioni della componente di entità non elevata di emotorace destro, moderato incremento della componente liquida di versamento associata. Persiste addensamento compressivo del parenchima adiacente. Lieve incremento del versamento pleurico sinistro, sempre di entità non elevata, ed associato a fenomeni di disventilazione compressiva del parenchima adiacente. Non sanguinamenti in atto. Nettamente ridotto lo scollamento pleurico destro, permane minimo in sede basale anteriore. Invariato il resto. ADDOME: non modificazioni significative degli ematomi in sede epatica ed al polo renale superiore destro. Non sanguinamenti in atto. Lieve incremento dell'entità del versamento liquido periepatico, lungo la doccia parietocolica destra, nello scavo pelvico ed in sede perisplenica (invariata la minima componente ematica in fase di organizzazione nel contesto del versamento pelvico). Non sovradistensione delle anse intestinali, che hanno pareti di normali spessore ed enhancement. Invariato il resto."


#bert_dir="/home/fusco/bt/model_custom"  
#tokenizer = AutoTokenizer.from_pretrained(bert_dir)  
#e=tokenizer.encode(ref, add_special_tokens=True)
#token = tokenizer.convert_ids_to_tokens(e)

#for t in token:
# print(t)


#G = nx.from_numpy_matrix(df, create_using=nx.DiGraph)
#layout = nx.spring_layout(G)
#nx.draw(G, layout)
#nx.draw_networkx_edge_labels(G, pos=layout)
#plt.show()




