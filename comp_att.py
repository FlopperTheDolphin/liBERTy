import torch
import transformers
import numpy as np
import pandas as pd

from vs_constants import *
from loader import *

def comp_matrix(tokenizer,model,sent_path):
 print('> '+ MTX_CAL)
 sentence,id_sent = load_sentence(sent_path)
 e=tokenizer.encode(sentence, add_special_tokens=True)
 output=model(torch.tensor([e]))
 #Matrix to load
 attentions = output[3]
 #tokens from sentence
 tokens = tokenizer.convert_ids_to_tokens(e)
 print('> '+ MTX_COMP)
  
 return attentions,tokens,id_sent
 
  
def save_matrix(dir_path,tokens,id_sent,attentions,verbose = True):
 # dimension (12,1,12,n,n)
 print('> ' + MTX_SAVE)
 for i in range(12):
  for j in range(12):
   np_attention = attentions[i][0][j].detach().numpy()
   df = pd.DataFrame(data=np_attention,columns=tokens)
   file_name = dir_path+"/"+ id_sent+ "_layer-"+str(i+1)+"_head-"+str(j+1)+".csv"
   df.to_csv(file_name, index=False)
   if(verbose) :
    print('> [' + file_name + '] Saved')
 print('> '+ MTX_SAVE_COMP)   


