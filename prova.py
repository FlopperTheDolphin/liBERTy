import torch
import transformers
import numpy as np
import pandas as pd
from transformers import AutoModel,AutoTokenizer, BertConfig
################################################################
 
#PRE_MATRIX
###########################################################################################  
bert_dir="/home/fusco/bt/model_custom"
config = BertConfig.from_pretrained(bert_dir, output_hidden_states=True,output_attentions=True)

tokenizer = AutoTokenizer.from_pretrained(bert_dir)
BERT = AutoModel.from_pretrained(bert_dir,config=config)



ref = "TC TORACE-ADDOME Esame condotto prima e dopo somministrazione di mdc endovena (Xenetix 350, 120 ml), confrontato con precedente del 12/11/2018. TORACE: rimossi la cannula tracheale ed il drenaggio toracico destro. Non significative modificazioni della componente di entità non elevata di emotorace destro, moderato incremento della componente liquida di versamento associata. Persiste addensamento compressivo del parenchima adiacente. Lieve incremento del versamento pleurico sinistro, sempre di entità non elevata, ed associato a fenomeni di disventilazione compressiva del parenchima adiacente. Non sanguinamenti in atto. Nettamente ridotto lo scollamento pleurico destro, permane minimo in sede basale anteriore. Invariato il resto. ADDOME: non modificazioni significative degli ematomi in sede epatica ed al polo renale superiore destro. Non sanguinamenti in atto. Lieve incremento dell'entità del versamento liquido periepatico, lungo la doccia parietocolica destra, nello scavo pelvico ed in sede perisplenica (invariata la minima componente ematica in fase di organizzazione nel contesto del versamento pelvico). Non sovradistensione delle anse intestinali, che hanno pareti di normali spessore ed enhancement. Invariato il resto."

e=tokenizer.encode(ref, add_special_tokens=True)

output=BERT(torch.tensor([e]))

last_hidden_state = output[0]
pooler_output = output[1]
hidden_states = output[2]
attentions = output[3]

print(len(hidden_states))
print(attentions[0].size())

token = tokenizer.convert_ids_to_tokens(e)
print(len(token))


#first hidden states = input of embedded words (1,n,768)
#last hidden states = output 
#attentions = our all heads attentions matrix (1,12,284,284)


#crea un csv per ogni singola head in questo modo è più rapido nella visualizzazione successiva


# dimension (12,1,12,n,n)
for i in range(12):
 for j in range(12):
  np_attention = attentions[i][0][j].detach().numpy()
  df = pd.DataFrame(data=np_attention,columns=token)
  df.to_csv("pr/layer-"+str(i+1)+"_head-"+str(j+1)+".csv", index=False)


#VISUALIZZAZIONE
df = pd.read_csv("pr/layer-1_head-1.csv")

print(df)


# for i in range(12):
#  np.savetxt("foo.csv", np_attention[0][i], delimiter=",")



#import csv 

#filename = 'p.csv'
    

#with open(filename, 'w') as csvfile: 
 #   csvwriter = csv.writer(csvfile) 
   # csvwriter.writerow(fields) 
  #  csvwriter.writerows(attentions)


#    
#with open('Giants.csv', mode ='r')as file: 
#  csvFile = csv.reader(file) 
#  for lines in csvFile: 
#        print(lines) 



