import sys
import os
import getopt
import gc
from configparser import ConfigParser
from comp_att import *
from loader import *
from view import *

# save matrix in a file
def load(sent_id): 
  
 tokenizer,model = load_model(model_dir,model_dir)
 sent_path = os.path.join(sent_dir, sent_id)
 attentions,tokens,id_sent = comp_matrix(tokenizer,model,sent_path)
 path = os.path.join(out_dir, sent_id)
 
 if(not os.path.isdir(path)): 
  os.mkdir(path)
 save_matrix(path,tokens,attentions)


def tokens(sent_id):
 tokenizer =  load_tokenizer(model_dir)
 sent_path = os.path.join(sent_dir, sent_id) + ".xml"
 print(comp_token(tokenizer,sent_path))

def see(sent_id,layer,head,token):
 
  fram = select_sub_matrix_for_token(out_dir,sent_id,layer,head,token)
  tokenizer =  load_tokenizer(model_dir)
  sent_path = os.path.join(sent_dir, sent_id) + ".xml"
  tokens = comp_token(tokenizer,sent_path)
    
  #tokens is used to associate number of row to token
  
  view_token(fram,tokens,token)
  
  

def main():
	
##PARSE 
 
 parser = ConfigParser()
 parser.read("config.txt")
 
 global model_dir, out_dir, sent_dir  
 model_dir = parser.get("config", "model_dir") 
 out_dir = parser.get("config", "out_dir")
 sent_dir = parser.get("config", "sent_dir")
 

 main_option = sys.argv[1]


 argument_list = sys.argv[2:]
 options = "s:l:h:t:"
 long_options = ["sentence","layer","head","token"]
 
 try:
    arguments, values = getopt.getopt(argument_list, options, long_options)
    if  main_option in ("h", "help","-h","--help"):
        print ("Help")
             
    elif main_option == "load":
          if arguments[0][0] != None and arguments[0][0] in ("-s","--sentence"):
           load(arguments[0][1])
          else:
           print("> ERROR: require id sentence")
          
     
    elif main_option == "see_token":
         for current_arg in arguments:
        
          if current_arg[0] in ("-s","--sentence"):
           sent_id = current_arg[1]
          elif current_arg[0] in ("-l","--layer"):
           layer= current_arg[1]
          elif current_arg[0] in ("-h","--head"):
           head = current_arg[1]
          elif current_arg[0] in ("-t","--token"):
           token = current_arg[1]
          else:
           print("> option not recognized")  
          
         see_token(sent_id,layer,head,token)
    
    elif main_option == "tokens":
      if arguments[0][0] != None and arguments[0][0] in ("-s","--sentence"):
       tokens(arguments[0][1])
      else:
       print("> ERROR: require id sentence")
     
    
    else:
       print("> no command here")        
 except getopt.error as err:
   print (str(err))



#PARSE THINGS TODO


if __name__ == '__main__':
    main()
    sys.exit()

