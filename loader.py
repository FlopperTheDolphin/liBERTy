import sys
import os
import json
#from pp.pre_processing import extract_text_from_request
import xml.etree.ElementTree as ET
import requests
import pandas as pd
#from bs4 import BeautifulSoup
from os import listdir
from vs_constants import * 
from transformers import AutoModel,AutoTokenizer, BertConfig


def get_ref_from_xml(ref_xml_path,name_file):
 files = listdir(ref_xml_path)
 for f in files:
   if(str(f) == str(name_file)):
        xml_file = open(ref_xml_path+'/'+f, 'r')
        contents = xml_file.read()
        soup = BeautifulSoup(contents, "xml")
        break  	
 return soup.find('REFERTO').get_text()

def load_model (bert_path,tok_path) :

 config = BertConfig.from_pretrained(bert_path, output_hidden_states=True,output_attentions=True)
 #print('> ' + TKN_LOADING)
 tokenizer = load_tokenizer(bert_path)
 #print('> '+ TKN_LOADED)
 print('> '+ MDL_LOADING)
 model = AutoModel.from_pretrained(bert_path,config=config)
 print('> '+ MDL_LOADED)
 
 return tokenizer,model

#def load_sentece(sent_xml_path,name_file):
# sentence = get_ref_from_xml(sent_xml_path,name_file)       
# print('> '+ SNT_LOADED)
# return sentence

#def load_sentence(name_file):
 #return "TC TORACE-ADDOME Esame condotto prima e dopo somministrazione di mdc endovena (Xenetix 350, 120 ml), confrontato con precedente del 12/11/2018. TORACE: rimossi la cannula tracheale ed il drenaggio toracico destro. Non significative modificazioni della componente di entità non elevata di emotorace destro, moderato incremento della componente liquida di versamento associata. Persiste addensamento compressivo del parenchima adiacente. Lieve incremento del versamento pleurico sinistro, sempre di entità non elevata, ed associato a fenomeni di disventilazione compressiva del parenchima adiacente. Non sanguinamenti in atto. Nettamente ridotto lo scollamento pleurico destro, permane minimo in sede basale anteriore. Invariato il resto. ADDOME: non modificazioni significative degli ematomi in sede epatica ed al polo renale superiore destro. Non sanguinamenti in atto. Lieve incremento dell'entità del versamento liquido periepatico, lungo la doccia parietocolica destra, nello scavo pelvico ed in sede perisplenica (invariata la minima componente ematica in fase di organizzazione nel contesto del versamento pelvico). Non sovradistensione delle anse intestinali, che hanno pareti di normali spessore ed enhancement. Invariato il resto.",'prova'

def load_matrix(out_dir,id_sent,layer,head):
 #file_dir = os.path.join(out_dir,id_sent) 
 file_path = out_dir+"/" + id_sent+ "/"+ "att-mtx_layer-"+layer+"_head-"+head+".csv"
 
 return pd.read_csv(file_path) 
 
def load_tokenizer(bert_path):
 return AutoTokenizer.from_pretrained(bert_path)
  
def load_from_json(name,path_file):

 f = open(path_file,'r')
 data = json.load(f)
 
 
 f.close()
 return data 

def save_in_json(name,path_file):
 out_file = open(path_file, "w") 
 json.dump(name, out_file, indent = 6)    
 out_file.close() 

