import spacy
from transformers import AutoModel,AutoTokenizer, BertConfig
ref = "TC TORACE-ADDOME Esame condotto prima e dopo somministrazione di mdc endovena (Xenetix 350, 120 ml), confrontato con precedente del 12/11/2018. TORACE: rimossi la cannula tracheale ed il drenaggio toracico destro. Non significative modificazioni della componente di entità non elevata di emotorace destro, moderato incremento della componente liquida di versamento associata. Persiste addensamento compressivo del parenchima adiacente. Lieve incremento del versamento pleurico sinistro, sempre di entità non elevata, ed associato a fenomeni di disventilazione compressiva del parenchima adiacente. Non sanguinamenti in atto. Nettamente ridotto lo scollamento pleurico destro, permane minimo in sede basale anteriore. Invariato il resto. ADDOME: non modificazioni significative degli ematomi in sede epatica ed al polo renale superiore destro. Non sanguinamenti in atto. Lieve incremento dell'entità del versamento liquido periepatico, lungo la doccia parietocolica destra, nello scavo pelvico ed in sede perisplenica (invariata la minima componente ematica in fase di organizzazione nel contesto del versamento pelvico). Non sovradistensione delle anse intestinali, che hanno pareti di normali spessore ed enhancement. Invariato il resto."


bert_dir="/home/fusco/bt/model_custom"


tokenizer = AutoTokenizer.from_pretrained(bert_dir)
e=tokenizer.encode(ref, add_special_tokens=True)
tk = tokenizer.convert_ids_to_tokens(e)


nlp = spacy.load("it_core_news_sm")
doc = nlp(ref)



#for t in tk:
 #print(len(t))
 
#print(str(len(tk)) + " " + str(len(doc)))

  
w=list()
  
  

def labelizer(l1,l2,dic=None):
 if dic == None:
  dic = dict()
  
 if len(l2)==0 or len(l1)==0:
  return dic
 
 obj_v = l2.pop(0)
 v = str(obj_v)
 #obj_v troviamo tutto l'oggetto ma a noi ci serve anche la sua stringa
 #per il confronto
  
 s = l1.pop(0)
 
 if(s == '[CLS]'):
  s=l1.pop(0)
 
 if(s == '[SEP]'):
  return dic
 
 k=list()
 k.append(s)
 
 while(v != s and len(l1) != 0):
  t = l1.pop(0)
  k.append(t)
  t= t.replace('##','') 
  s = s+t 
 
 for token in k:
  dic[token] = obj_v.pos_ 
  print('caricato token ' + token + ' con pos ' + str(obj_v.pos_) )
  
 print(str(l1))
 print(str(l2))
 
 labelizer(l1,l2,dic)
 return dic   



for token in doc:
 w.append(token)

q=list()+w

l3=labelizer(tk,q)

