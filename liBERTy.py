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
def load(): 
  
 tokenizer,model = load_model(model_dir,model_dir)
 #sent_path = os.path.join(sent_dir, sent_id)
 attentions,tokens = comp_matrix(tokenizer,model,sentence)
 path = os.path.join(out_dir, sent_id)
 
 if(not os.path.isdir(path)): 
  os.mkdir(path)
 save_matrix(path,tokens,attentions)


def tokens():
 tokenizer =  load_tokenizer(model_dir)
 #sent_path = os.path.join(sent_dir, sent_id) + ".xml"
 print(comp_token(tokenizer,sentence))

def see_att_token(layer,head,word,sent,time=None):
 
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
  
  dic_sent = tokens_to_sentence(bt,sp)
  
  #fram_dic = dict()
  
  for tk in tokens:
   if dic_sent[tk] == word:
    print('TOKEN: ' +tk)
    print('') 
    fram = select_sub_matrix_for_token(out_dir,sent_id,layer,head,tk)
    
    if sent == True:
     view_sentence_perc(fram,tokens,tk)
     if time != None: 
      view_sentence_time(fram,tokens,tk,time)
     view_word(fram,tokens,tk,time) 
    else:
     view_word(fram,tokens,tk,time)
 
   
def see_pos(layer,head):
  
  #sent_path = os.path.join(sent_dir, sent_id)
  #sentence,id_sent = load_sentence(sent_path)
  
  nlp = spacy.load("it_core_news_sm")
  doc = nlp(sentence)
  
  spacy_token=list()
  for token in doc:
   spacy_token.append(token)

  tokenizer = load_tokenizer(model_dir)
  #sent_path = os.path.join(sent_dir, sent_id) + ".xml"
  bert_token = comp_token(tokenizer,sentence)
  
  bt = list() + bert_token
  
  dic_tokens = labelizer(bt,spacy_token)
  
  att_mtx = load_matrix(out_dir,sent_id,layer,head)
  
  dic_pos,dic_edge = get_pos_mtx(att_mtx,dic_tokens,bert_token)
  
  view_mtx_pos(dic_edge,dic_pos)  

def main():
	
##PARSE 
 
 parser = ConfigParser()
 parser.read("config.txt")
 
 global model_dir, out_dir, sent_dir, sentence,sent_id  
 model_dir = parser.get("config", "model_dir") 
 out_dir = parser.get("config", "out_dir")
 sent_dir = parser.get("config", "sent_dir")
 sentence = parser.get("config","sentence") 
 sent_id = parser.get("config","sentence_id")
 
 main_option = sys.argv[1]


 argument_list = sys.argv[2:]
 options = "l:h:w:st:"
 long_options = ["layer","head","word","sentence","time"]
 
 try:
    arguments, values = getopt.getopt(argument_list, options, long_options)
    if  main_option in ("h", "help","-h","--help"):
        print ("load - load matrix's sentence")
        print("see [-l | --layer = <layer>] [-h | --head = <head>] [-w | --word = <word>] [-s | --sentence] [-t | --time = <time>] - show most weighted token given a word: sentence = change style of visualisation, time = select how many tokens correlated find")
        print("pos [-l | --layer = <layer>] [-h | --head = <head>] - show the pos graph given a attention matrix")
        print("tokens - print all sentence tokens")
        print("sentence - print the sentence")
             
    elif main_option == "load":
          #if arguments[0][0] != None and arguments[0][0] in ("-s","--sentence"):
          load()
          #else:
           #print("> ERROR: require id sentence")
          
     
    elif main_option == "see":
    
         sent = False
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
          else:
           print("> option not recognized")  
          
                   
         see_att_token(layer,head,word,sent,time)
    
    elif main_option == "tokens":
      #if arguments[0][0] != None and arguments[0][0] in ("-s","--sentence"):
       tokens()
      #else:
      # print("> ERROR: require id sentence")
    
    elif main_option == "sentence":
     print(sentence)
      
    elif main_option == "pos":
    
        for current_arg in arguments:
 
          #if current_arg[0] in ("-s","--sentence"):
           #sent_id = current_arg[1]
          if current_arg[0] in ("-l","--layer"):
           layer= current_arg[1]
          elif current_arg[0] in ("-h","--head"):
           head = current_arg[1]
          else:
           print("> option not recognized")  
        
        see_pos(layer,head)
                
                
     #elif main_option == "total_view":
                
    else:
       print("> no command here")        
 except getopt.error as err:
   print (str(err))



#PARSE THINGS TODO


if __name__ == '__main__':
    main()
    sys.exit()

