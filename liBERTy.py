import sys
import os
import getopt
from comp_att import *
from loader import *


# save matrix in a file
def load():
 model_path = sys.argv[1]
 sent_path = sys.argv[2]
 out_dir = sys.argv[3]

 tokenizer,model = load_model(model_path,model_path)

 attentions,tokens,id_sent = comp_matrix(tokenizer,model,sent_path)

 path = os.path.join(out_dir, id_sent)

 if(not os.path.isdir(path)): 
  os.mkdir(path)

 save_matrix(path,tokens,id_sent,attentions)


def see():
  sent_path = sys.argv[1]
  out_dir = sys.argv[2]




def main():
 see()

#PARSE THINGS TODO

#argument_list = sys.argv[1:]
#options = "lshmso:"
#long_options = ['load',"see"]



#try:
 #   arguments, values = getopt.getopt(argument_list, options, long_options)
  #   
#
 #   for currentArgument, currentValue in arguments:
 #
  #      if currentArgument in ("-h", "--Help"):
   #        print ("Displaying Help")
    #         
    #    elif currentArgument in ("-l", "--load"):
    #        print ("Scelto load")
    #        load()
#
             
 #       elif currentArgument in ("-s", "--see"):
  #          print ("Scelto see")
             
#except getopt.error as err:
 #   print (str(err))

if __name__ == '__main__':
    main()

