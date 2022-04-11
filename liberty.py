import sys
import getopt
from helper import *


     
def main():
	
 model_dir, out_dir,ref_dir= read_config_file() 

 try:
  name = read_config_att("config.txt","name")
 except Exception:
  name = None
   
 main_option = sys.argv[1]


 argument_list = sys.argv[2:]
 options = "f:i:n:v:t:"
 long_options = ["file","id","name","vector"]

 try:
    arguments, values = getopt.getopt(argument_list, options, long_options)
    if  main_option in ("h", "help","-h","--help"):
      total_help()  
             
    elif main_option == "load":
     file_path = None
     name = None
      
     for current_arg in arguments:
      if current_arg[0] in ("-f","--file"):
       file_path = current_arg[1]
      elif current_arg[0] in ("-n","--name"):
       name= current_arg[1]
       
     if file_path == None or name == None:
      print(help('load'))
      sys.exit()
     
     sentence,sent_id = read_sentence(file_path)        
     from features.load import load,load_score               
     load(sentence,sent_id,name,model_dir, out_dir)
     load_score(model_dir,out_dir,name,sentence)                
      
    elif main_option == "tokens":
     if len(arguments) > 1 and arguments[0][0] in ("-n","--name"):
      name = arguments[0][1]
     from features.utiliy import tokens 
     tokens(name,out_dir,model_dir)
    
    elif main_option == "sentence":
     if len(arguments) != 0 and arguments[0][0] in ("-n","--name"):
      name = arguments[0][1]
     from features.utiliy import get_sentence 
     print(get_sentence(out_dir,name))
     
     
    elif main_option == "stat":
     token=None
     vector = None
     id_token=0
     
     for current_arg in arguments:
       if current_arg[0] in ("-t","--token"):
            token= current_arg[1]
       elif current_arg[0] in ("-n","--name"):
            name= current_arg[1]
       elif current_arg[0] in ("-v","--vector"):
            vector= current_arg[1]     
       elif current_arg[0] in ("-i","--id"):
            id_token= current_arg[1]     
                                                             
     if token == 'all' and vector != None:
      from features.stat import total_stat
      total_stat(name,out_dir,model_dir,vector)
      sys.exit()
     elif token != None and vector != None:                   
      from features.stat import see_stat
      see_stat(name,out_dir,token,model_dir,vector,id_token)
       
     else: 
      print(help('stat'))      
      sys.exit()
    
    else:
       print("> no command here")        
 except getopt.error as err:
   print (str(err))


if __name__ == '__main__':
    main()
    sys.exit()

