from configparser import ConfigParser
import sys
import getopt
from helper import *
#from warmup import *

#read sentence and sentence id from correct file
def read_sentence(file_path):
  parser = ConfigParser()
  parser.read(file_path)
  sentence = parser.get("sentence","sentence") 
  sent_id = parser.get("sentence","sentence_id")
  return sentence,sent_id
 
def read_config_file():
 parser = ConfigParser()
 parser.read("config.txt")   
 model_dir = parser.get("config", "model_dir") 
 out_dir = parser.get("config", "out_dir")
 sent_dir = parser.get("config", "sent_dir")
 return model_dir, out_dir, sent_dir


def read_config_att(file_path,n):
 parser = ConfigParser()
 parser.read(file_path)
 return parser.get("config",str(n))
     
def main():
	
##PARSE 
 model_dir, out_dir, sent_dir = read_config_file() 

 try:
  name = read_config_att("config.txt","name")
 except Exception:
  name = None
   
 main_option = sys.argv[1]


 argument_list = sys.argv[2:]
 options = "l:h:w:sct:f:i:n:p:r:v:"
 long_options = ["layer","head","word","sentence","time","class","file","id","name","perc","vector"]
 
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
     sentence,sent_id = read_sentence(current_arg[1])        
     from features.load import load               
     load(sentence,sent_id,name,model_dir, out_dir)
           
          #else:
           #print("> ERROR: require id sentence")
          
    elif main_option == "see":
     layer=None
     head=None
     word=None
     time=None
     for current_arg in arguments:   
      if current_arg[0] in ("-l","--layer"):
       layer= current_arg[1]
      elif current_arg[0] in ("-h","--head"):
       head = current_arg[1]
      elif current_arg[0] in ("-w","--word"):
       word = current_arg[1]
      elif current_arg[0] in ("-t","--time","t"):
       time = current_arg[1] 
      elif current_arg[0] in ("-n","--name"):
       name = current_arg[1] 
      else:
       print("> option not recognized")
     if layer==None or head == None or word== None:
      print(help('see'))
      sys.exit()    
     from features.see_attention import see_attention_sentence       
     see_attention_sentence(name,layer,head,word,out_dir,model_dir,time)
      
         
    elif main_option == "tokens":
     if len(arguments) > 1 and arguments[0][0] in ("-n","--name"):
      name = arguments[0][1]
     from features.utiliy import tokens 
     tokens(name,out_dir,model_dir)
    
       
      #else:
      # print("> ERROR: require id sentence")
    
    elif main_option == "sentence":
     if len(arguments) != 0 and arguments[0][0] in ("-n","--name"):
      name = arguments[0][1]
     from features.utiliy import get_sentence 
     print(get_sentence(out_dir,name))
     
     
    elif main_option == "stat":
     token=None
     perc=None
     vector = None
     id_token=0
     
     for current_arg in arguments:
       if current_arg[0] in ("-t","--token"):
            token= current_arg[1]
       elif current_arg[0] in ("-n","--name"):
            name= current_arg[1]
       elif current_arg[0] in ("-p","--perc"):
            perc= current_arg[1]
       elif current_arg[0] in ("-v","--vector"):
            vector= current_arg[1]     
       elif current_arg[0] in ("-i","--id"):
            id_token= current_arg[1]     
                        
     if token == 'all' and vector != 'all' and vector != None:
      from features.stat import total_stat
      total_stat(name,out_dir,model_dir,vector)
      sys.exit()
     elif token != None and perc == None:                   
      try:       
       from features.stat import see_stat
       see_stat(name,out_dir,token,model_dir,vector,id_token)
      except IndexError:
       from features.utiliy import tokens 
       from features.utiliy import find
       from features.utiliy import get_sentence
       bert_tokens = tokens(name,out_dir,model_dir,view=False)       
       sentence = get_sentence(out_dir,name)     
       find(sentence,bert_tokens,token)
     elif perc != None: 
      from features.stat import comp_stat       
      comp_stat(name,out_dir,perc,model_dir,vector)
     else: 
      print(help('stat'))      
      sys.exit()
    
    
    elif main_option == "outlier":
       layer=None
       head = None 
       vector= None
       for current_arg in arguments:
        if current_arg[0] in ("-n","--name"):
         name = current_arg[1]
        elif current_arg[0] in ("-l","--layer"):
         layer= current_arg[1]
        elif current_arg[0] in ("-h","--head"):
         head = current_arg[1]
        elif current_arg[0] in ("-v","--vector"):
         vector = current_arg[1]   
        else:
         print("> option not recognized")
         
       if (layer=='all' or head == 'all') and vector != None:
        from features.stat import total_outlier
        total_outlier(name,out_dir,model_dir,vector)
        sys.exit()
          
       if layer == None or head == None or vector == None:
        print(help('outlier'))
        sys.exit()        
       from features.stat import outlier     
       outlier(name,layer,head,out_dir,model_dir,vector) 
           
        
    elif main_option == "pos":
       layer=None
       head = None
       for current_arg in arguments:
        if current_arg[0] in ("-n","--name"):
         name = current_arg[1]
        elif current_arg[0] in ("-l","--layer"):
         layer= current_arg[1]
        elif current_arg[0] in ("-h","--head"):
         head = current_arg[1]
        else:
         print("> option not recognized")
           
       if layer == None or head == None:
        print(help('pos'))
        sys.exit()        
       from features.pos import see_pos     
       see_pos(name,layer,head,out_dir,model_dir)
                          
   # elif main_option == "see_total":
    #   for current_arg in arguments:
 
#          if current_arg[0] in ("-n","--name"):
 #          name = current_arg[1]
  #        elif current_arg[0] in ("-l","--layer"):
   #        layer= current_arg[1]
    #     elif current_arg[0] in ("-h","--head"):
     #      head = current_arg[1]
      #    else:
      #     print("> option not recognized")  
      # see_total(name,layer,head) 
      
    elif main_option == "who":
       layer=None
       head=None
       per=None
       for current_arg in arguments:
          if current_arg[0] in ("-n","--name"):
           name = current_arg[1]
          elif current_arg[0] in ("-l","--layer"):
           layer= current_arg[1]
          elif current_arg[0] in ("-h","--head"):
           head = current_arg[1]
          elif current_arg[0] in ("-p","--perc"):
           perc = current_arg[1] 
          else:
           print("> option not recognized")
       if layer==None or head==None or perc==None:      
        print(help('who'))
        sys.exit()
       from features.stat import who    
       who(name,layer,head,perc,out_dir,model_dir)
       
    elif main_option == "dist":
     for current_arg in arguments:
        if current_arg[0] in ("-n","--name"):
          name = current_arg[1] 
        else:
          print("> option not recognized")       
     from features.distance import distance  
     distance(name,out_dir)
     
    elif main_option == "cluster":
     n_centroid = None
     for current_arg in arguments:
        if current_arg[0] in ("-n","--name"):
          name = current_arg[1] 
        elif current_arg[0] in ("-r","--centroids"):
          n_centroid = current_arg[1] 
        else:
          print("> option not recognized")
     if n_centroid == None:
      print(help('cluster'))
      sys.exit()      
     from features.distance import cluster     
     cluster(name,out_dir,model_dir,n_centroid)
     
    elif main_option == "smear":
     layer=None
     head=None
     token=None
     for current_arg in arguments:
        if current_arg[0] in ("-n","--name"):
          name = current_arg[1] 
        elif current_arg[0] in ("-t","--token"):
          token = current_arg[1]
        elif current_arg[0] in ("-l","--layer"):
          layer = current_arg[1]
        elif current_arg[0] in ("-h","--head"):
          head = current_arg[1]    
        else:
          print("> option not recognized")  
    
    
    elif main_option == "find":
     token=None
     for current_arg in arguments:
        if current_arg[0] in ("-n","--name"):
          name = current_arg[1] 
        elif current_arg[0] in ("-t","--token"):
          token = current_arg[1]    
        else:
          print("> option not recognized")  
    
    
     if token == None:
      print(help('find'))     
      sys.exit()
     from features.utiliy import tokens 
     from features.utiliy import find
     from features.utiliy import get_sentence
     bert_tokens = tokens(name,out_dir,model_dir,view=False)       
     sentence = get_sentence(out_dir,name)     
     find(sentence,bert_tokens,token)
    
    #elif main_option == "total_stat":
     #if arguments[0][0] != None and arguments[0][0] in ("-n","--name"):
     # name = arguments[0][1]
     #total_stat(name,out_dir,model_dir) 
    
 #   elif main_option == "noop":
 #    for current_arg in arguments:
 #       if current_arg[0] in ("-n","--name"):
 #         name = current_arg[1] 
 #       elif current_arg[0] in ("-l","--layer"):
  #        layer = current_arg[1]
  #      elif current_arg[0] in ("-h","--head"):
  #        head = current_arg[1]
  #   noop_search(name,out_dir,model_dir,layer,head) 
    
  #  elif main_option == "total_noop":
  #   for current_arg in arguments:
  #      if current_arg[0] in ("-n","--name"):
  #        name = current_arg[1] 
          
  #   total_noop(name,out_dir,model_dir) 
    
    
   # elif main_option == "compare":
 #    for current_arg in arguments:
 #       if current_arg[0] in ("-n","--name"):
 #         name = current_arg[1] 
          
  #   compare(name,out_dir,model_dir) 
     
    else:
       print("> no command here")        
 except getopt.error as err:
   print (str(err))

#PARSE THINGS TODO

if __name__ == '__main__':
    main()
    sys.exit()

