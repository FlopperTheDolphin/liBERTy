from configparser import ConfigParser

from fun.vs_constants import SEPARATOR
def help(command):
 if command == 'load':
  return "load [-f | --file = <path to file sentence>] [-n | --name = <name of sentence>]"
 
 elif command == 'stat':
  return "stat [-t | --token = <token to visualise>] {-n | --name = <name of the sentence>} [-v | --vector = <vector used for shannon>] {-i | --id = <number of token>} - print all the Shannon distance from a specified token for all the matrix from a specified vector,\n * define which vectore use among: entropy,noop and me.\n * If token == all plot statistic values (avg and std)"
 elif command == 'tokens':
  return "tokens - print all sentence tokens"   
 elif command == 'sentence':
  return "sentence - print the sentence"   
  
def total_help():
 print('* ' + help('load'))
 print()

 print('* ' + help('stat'))
 print()
 
 print()
 print('* ' + help('tokens'))
 print()
 print('* ' + help('sentence'))
 print()
 print('* ' + help('find'))
 print()
 print('* ' + help('outlier'))   
 
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
 ref_dir = parser.get("config","ref_dir")
 #sent_dir = parser.get("config", "sent_dir")
 return model_dir, out_dir,ref_dir#, sent_dir


def read_config_att(file_path,n):
 parser = ConfigParser()
 parser.read(file_path)
 return parser.get("config",str(n))
 
