from configparser import ConfigParser
from features.load import load
from features.see_attention import see_attention_sentence,smear,noop_search,total_noop
from features.stat import see_stat,comp_stat,who,total_stat
from features.pos import see_pos  
from features.utiliy import tokens
from features.distance import distance,cluster
import sys
import getopt

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
 options = "l:h:w:sct:f:i:n:p:r:"
 long_options = ["layer","head","word","sentence","time","class","file","id","name","perc"]
 
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
               
          load(sentence,sent_id,name,model_dir, out_dir)
          #else:
           #print("> ERROR: require id sentence")
          
    elif main_option == "see":
    
         sent = False
         cls = False
         time=None
         for current_arg in arguments:
        
          if current_arg[0] in ("-l","--layer"):
           layer= current_arg[1]
          elif current_arg[0] in ("-h","--head"):
           head = current_arg[1]
          elif current_arg[0] in ("-w","--word"):
           word = current_arg[1]
          #elif current_arg[0] in ("-s","--sentence","s"):
           #sent = True 
          elif current_arg[0] in ("-t","--time","t"):
           time = current_arg[1] 
          #elif current_arg[0] in ("-c","--class","c"):
           #cls = True
          elif current_arg[0] in ("-n","--name"):
           name = current_arg[1] 
          else:
           print("> option not recognized")  
          
         see_attention_sentence(name,layer,head,word,True,True,out_dir,model_dir,time)
    
    elif main_option == "tokens":
      #if arguments[0][0] != None and arguments[0][0] in ("-n","--name"):
       #name = arguments[0][1]
       
      tokens(name,out_dir,model_dir)
       
      #else:
      # print("> ERROR: require id sentence")
    
    elif main_option == "name":
      #if arguments[0][0] != None and arguments[0][0] in ("-n","--name"):
       ##TODO
       print('')
    
    elif main_option == "sentence":
     if arguments[0][0] != None and arguments[0][0] in ("-n","--name"):
      name = arguments[0][1]
     
     sentence(out_dir,name)
      
    elif main_option == "stat":
     for current_arg in arguments:
      if current_arg[0] in ("-t","--token"):
            token= current_arg[1]
      elif current_arg[0] in ("-n","--name"):
            name= current_arg[1]            
             
     see_stat(name,out_dir,token,model_dir)
     
    elif main_option== 'comp_stat':
      for current_arg in arguments:
       if current_arg[0] in ("-p","--perc"):
            perc= current_arg[1]
       elif current_arg[0] in ("-n","--name"):
            name= current_arg[1]
             
      comp_stat(name,out_dir,perc,model_dir)
     
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
       who(name,layer,head,perc,out_dir,model_dir)
       
    elif main_option == "dist":
     for current_arg in arguments:
        if current_arg[0] in ("-n","--name"):
          name = current_arg[1] 
        else:
          print("> option not recognized")  
        distance(name,out_dir)
     
    elif main_option == "cluster":
     for current_arg in arguments:
        if current_arg[0] in ("-n","--name"):
          name = current_arg[1] 
        elif current_arg[0] in ("-r","--centroids"):
          n_centroid = current_arg[1] 
        else:
          print("> option not recognized")  
     cluster(name,out_dir,model_dir,n_centroid)
     
    elif main_option == "smear":
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
     smear(name,out_dir,model_dir,token,layer,head)
    
    elif main_option == "total_stat":
     #if arguments[0][0] != None and arguments[0][0] in ("-n","--name"):
      #name = arguments[0][1]
     total_stat(name,out_dir,model_dir) 
    
    elif main_option == "noop":
     for current_arg in arguments:
        if current_arg[0] in ("-n","--name"):
          name = current_arg[1] 
        elif current_arg[0] in ("-l","--layer"):
          layer = current_arg[1]
        elif current_arg[0] in ("-h","--head"):
          head = current_arg[1]
     noop_search(name,out_dir,model_dir,layer,head) 
    
    elif main_option == "total_noop":
     for current_arg in arguments:
        if current_arg[0] in ("-n","--name"):
          name = current_arg[1] 
          
     total_noop(name,out_dir,model_dir) 
    
    
    else:
       print("> no command here")        
 except getopt.error as err:
   print (str(err))

#PARSE THINGS TODO

if __name__ == '__main__':
    main()
    sys.exit()

