import sys
import os
import getopt
import gc
import spacy
from configparser import ConfigParser
from comp_att import *
from loader import *
from view import *


# save matrix in a file
def load(sentence,sent_id,name): 
  
 tokenizer,model = load_model(model_dir,model_dir)
 #sent_path = os.path.join(sent_dir, sent_id)
 attentions,tokens = comp_matrix(tokenizer,model,sentence)
 mtx_path = os.path.join(out_dir, name)
 max_path= os.path.join(mtx_path,"max.json")
 sent_path= os.path.join(mtx_path,"sentence.json")
 if(not os.path.isdir(mtx_path)): 
  os.mkdir(mtx_path)
 dic_sent = dict()
 dic_sent['sentence'] = sentence
 dic_sent['sent_id'] = sent_id
 save_in_json(dic_sent,sent_path)
 att_max = save_matrix(mtx_path,tokens,attentions)

def tokens(name):
 dic_sent = load_from_json(os.path.join(os.path.join(out_dir, name),'sentence.json'))
 sentence = dic_sent['sentence']
 tokenizer =  load_tokenizer(model_dir)
 #sent_path = os.path.join(sent_dir, sent_id) + ".xml"
 print(comp_token(tokenizer,sentence))

def see_total(name,layer,head):
  dic_sent = load_from_json(os.path.join(os.path.join(out_dir, name),'sentence.json'))
  sentence = dic_sent['sentence']
  tokenizer =  load_tokenizer(model_dir)
  tokens = comp_token(tokenizer,sentence)
  
  list_weight=comp_token_weight(tokens,layer,head,name,out_dir)
  
  view_att_total(name,list_weight,tokens)
  
  
   

def see_att_token(name,layer,head,word,sent,cls,time=None):
  dic_sent = load_from_json(os.path.join(os.path.join(out_dir, name),'sentence.json'))
  sentence = dic_sent['sentence']
  tokenizer =  load_tokenizer(model_dir)
  #sent_path = os.path.join(sent_dir, sent_id) + ".xml"
  tokens = comp_token(tokenizer,sentence)
  
  nlp = spacy.load("it_core_news_sm")
  doc = nlp(sentence)
  spacy_token=list()
  for token in doc:
   spacy_token.append(token)
   
  sp = list()+spacy_token 
  bt = list()+tokens 
  
  mtx_dir = os.path.join(out_dir, name)
  
  path_cache = os.path.join(mtx_dir,"token_sent.json") 
  
  if not os.path.exists(path_cache):
    os.mknod(path_cache)

  dic_sent = tokens_to_sentence(bt,sp,path_cache)
  
  for tk in tokens:
   if dic_sent[tk] == word:
    print('TOKEN: ' +tk)
    print('') 
    fram = select_sub_matrix_for_token(out_dir,name,layer,head,tk)
    
    if sent == True:
     max_path= os.path.join(mtx_dir,"max.json")
     mx= load_from_json(max_path)
     view_sentence_perc(fram,tokens,tk,layer,head,mx)
     if time != None: 
      view_sentence_time(fram,tokens,tk,time)
      
    if cls == True: 
     view_word(fram,tokens,tk,time)
      

def see_stat(name,token):
  print('frase: ' + name)
  dic_sent = load_from_json(os.path.join(os.path.join(out_dir, name),'sentence.json'))
  sentence = dic_sent['sentence']
  tokenizer =  load_tokenizer(model_dir)
  tokens = comp_token(tokenizer,sentence)        
  dic_att = get_all_att_sentece(out_dir,name,token)
  n_token = len(tokens)
  df,l=comp_divergence(dic_att,n_token)
  view_token_div(df,token,l)
     
   
def see_pos(name,layer,head):
  dic_sent = load_from_json(os.path.join(os.path.join(out_dir, name),'sentence.json'))
  sentence = dic_sent['sentence']
  path_graph = os.path.join(os.path.join(out_dir,name),"pos-graph_layer-"+str(layer)+"_head-"+str(head)+".json")
  
  if not os.path.exists(path_graph):
    os.mknod(path_graph)
    
  if(view_loaded_pos(path_graph) == True):
   return  
  
  nlp = spacy.load("it_core_news_sm")
  doc = nlp(sentence)
  
  spacy_token=list()
  for token in doc:
   spacy_token.append(token)

  #for token in doc:
   #print(str(token) + "---->"+str(token.pos_))
  
  tokenizer = load_tokenizer(model_dir)
  #sent_path = os.path.join(sent_dir, sent_id) + ".xml"
  bert_token = comp_token(tokenizer,sentence)
  
  bt = list() + bert_token
  
  path_cache = os.path.join(os.path.join(out_dir, name),"token_pos.json") 
  
  if not os.path.exists(path_cache):
    os.mknod(path_cache)
  
  dic_tokens = labelizer(bt,spacy_token,path_cache)
  
  att_mtx = load_matrix(out_dir,name,layer,head)
  
  dic_pos,dic_edge = get_pos_mtx(att_mtx,dic_tokens,bert_token)
  
  view_mtx_pos(dic_edge,dic_pos,path_cache,path_graph)  


def read_sentence(file_path):
  parser = ConfigParser()
  parser.read(file_path)
  sentence = parser.get("sentence","sentence") 
  sent_id = parser.get("sentence","sentence_id")
  return sentence,sent_id
 

def main():
	
##PARSE 
 
 parser = ConfigParser()
 parser.read("config.txt")
 
 global model_dir, out_dir, sent_dir   
 model_dir = parser.get("config", "model_dir") 
 out_dir = parser.get("config", "out_dir")
 sent_dir = parser.get("config", "sent_dir")
 #sentence = parser.get("config","sentence") 
 #sent_id = parser.get("config","sentence_id")
 
 main_option = sys.argv[1]


 argument_list = sys.argv[2:]
 options = "l:h:w:sct:f:i:n:"
 long_options = ["layer","head","word","sentence","time","class","file","id","name"]
 
 try:
    arguments, values = getopt.getopt(argument_list, options, long_options)
    if  main_option in ("h", "help","-h","--help"):
        print ("load - load matrix's sentence")
        print("see [-l | --layer = <layer>] [-h | --head = <head>] [-w | --word = <word>] [-s | --sentence] [-t | --time = <time>] - show most weighted token given a word: sentence = change style of visualisation, time = select how many tokens correlated find")
        print("pos [-l | --layer = <layer>] [-h | --head = <head>] - show the pos graph given a attention matrix")
        print("tokens - print all sentence tokens")
        print("sentence - print the sentence")
             
    elif main_option == "load":
          for current_arg in arguments:
           if current_arg[0] in ("-f","--file"):
            sentence,sent_id = read_sentence(current_arg[1])
           elif current_arg[0] in ("-n","--name"):
            name= current_arg[1]
          load(sentence,sent_id,name)
          #else:
           #print("> ERROR: require id sentence")
          
     
    elif main_option == "see":
    
         sent = False
         stat = False
         cls = False
         time=None
         for current_arg in arguments:
        
          if current_arg[0] in ("-l","--layer"):
           layer= current_arg[1]
          elif current_arg[0] in ("-h","--head"):
           head = current_arg[1]
          elif current_arg[0] in ("-w","--word"):
           word = current_arg[1]
          elif current_arg[0] in ("-s","--sentence"):
           sent = True 
          elif current_arg[0] in ("-t","--time"):
           time = current_arg[1] 
          elif current_arg[0] in ("-c","--class"):
           cls = True
          elif current_arg[0] in ("-n","--name"):
           name = current_arg[1] 
          else:
           print("> option not recognized")  
          
                   
         see_att_token(name,layer,head,word,sent,cls,time)
    
    elif main_option == "tokens":
      if arguments[0][0] != None and arguments[0][0] in ("-n","--name"):
       tokens(arguments[0][1])
      #else:
      # print("> ERROR: require id sentence")
    
    elif main_option == "sentence":
     if arguments[0][0] != None and arguments[0][0] in ("-n","--name"):
      dic_sent = load_from_json(os.path.join(os.path.join(out_dir, arguments[0][1]),'sentence.json'))
      print(dic_sent['sentence'])
     
    elif main_option == "stat":
    
     for current_arg in arguments:
      if current_arg[0] in ("-t","--token"):
            token= current_arg[1]
      elif current_arg[0] in ("-n","--name"):
            name= current_arg[1]      
            
     see_stat(name,token)
     
    elif main_option == "pos":
    
        for current_arg in arguments:
 
          if current_arg[0] in ("-n","--name"):
           name = current_arg[1]
          elif current_arg[0] in ("-l","--layer"):
           layer= current_arg[1]
          elif current_arg[0] in ("-h","--head"):
           head = current_arg[1]
          else:
           print("> option not recognized")  
        
        see_pos(name,layer,head)
                          
    elif main_option == "see_total":
       for current_arg in arguments:
 
          if current_arg[0] in ("-n","--name"):
           name = current_arg[1]
          elif current_arg[0] in ("-l","--layer"):
           layer= current_arg[1]
          elif current_arg[0] in ("-h","--head"):
           head = current_arg[1]
          else:
           print("> option not recognized")  
       see_total(name,layer,head) 
                
    else:
       print("> no command here")        
 except getopt.error as err:
   print (str(err))



#PARSE THINGS TODO


if __name__ == '__main__':
    main()
    sys.exit()
