from helper import *
from constants import *
from fun.loader import load_matrix
from features.utiliy import *
from features.see_attention import *
import spacy
import os
import sys
from fun.view import console_show,space
import pandas as pd 
from smear import smear
from gh import good_head


def gh():
 heads = cache['head']
 layers= cache['layer']
 
 for layer in layers:
  for head in heads:
   good_head(cache['name'],str(layer),str(head),out_dir,model_dir)
  

def see():
 heads = cache['head']
 layers= cache['layer']
       
 for layer in layers:
  for head in heads:
   see_attention_sentence(cache['name'],str(layer),str(head),token,out_dir,model_dir)
   
   if len(layers) > 2 or len(heads) > 2:
    res = input(INPUT)
    if res in ['exit','stop','bye']:
     break
      
def sme():
 heads = cache['head']
 layers= cache['layer']
 
 for layer in layers:
  for head in heads:
   print(layer,head)
   smear(head,layer,cache['name'],token,out_dir,model_dir,verbose=True)

def add_element_cache(name,value):
 cache[name]=value
   
def flush():
 k = dict()
 for key in cache.keys():
  if '(' not in key:
   k[key] = cache[key]
 
 return k  

def parse_select(cmds,i):
    subs = cmds[i+1].split(',')
    if len(subs) != 2:
     help()    
    else: 
     for j in range(2):
      if ':' in subs[j]:
       nums = [int(s) for s in subs[j].split(':') if s.isdigit()]
       if len(nums) > 2:
        console_show(MSG_ERROR_SYNTAX_1)
        break
       elif len(nums) == 0:
        console_show(ALL_LAYERS)
        nums = range(1,13,1)
       elif len(nums) == 2 and nums[0] < nums[1]:
        nums = range(nums[0],nums[1]+1,1)
       else:
        console_show(MSG_ERROR_SYNTAX_2)
        break
         
      else:
       nums = list()
       if subs[j].isdigit() == True:
        nums.append(subs[j])
       else:
        console_show(MSG_ERROR_SYNTAX_3)
        break
      print(nums)  
      if j == 0:
       layer = nums
       add_element_cache('layer',layer)
      else:
       head = nums
       add_element_cache('head',head)
           
       
def cache_contain(name):
 if cache[name] == None:
  return False
 else:
  return True 

def cache_contain_head():
 for key in cache.keys():
  if '(' in key:
   return True
 return False
   
def help():

 space()
 space()
 space()
 console_show(COMMAND,pick=False)
 console_show(SEE,pick=False)
 console_show(LH,pick=False) 
 console_show(GOOD,pick=False)  
 console_show(NAME,pick=False)
 console_show(BE,pick=False)
 space()
 space()
 space()
    
if __name__ == '__main__':
  
 cache=dict()
 repeat=True
 model_dir,out_dir,dir_ref = read_config_file()
 name = read_config_att("config.txt","name")
 add_element_cache('model_dir',model_dir)
 add_element_cache('out_dir',out_dir)
 add_element_cache('name',name)
 while(repeat):
  console_show(SEPARATOR,pick=False)
  console_show(TALKING_HEADS,pick=False)
  console_show('model:',cache['model_dir'],pick=False)
  console_show('sentence:',cache['name'],pick=False)
  if 'token' in cache.keys():
   console_show('TOKEN:',cache['token'],pick=False)
  if 'layer' in cache.keys() and 'head' in cache.keys():
   console_show('last head called','('+str(cache['layer'])+str(cache['head'])+')')
  console_show(SEPARATOR,pick=False)
  
  
  req = input(MSG_PRESS_CMD)
   
  cmds = req.split()
   
  if len(cmds) == 0:
   continue
  
  elif cmds[0] == 'select' or cmds[0] == 'sel':
     cache = flush()
     parse_select(cmds,0)
     
     layers = cache['layer']
     heads = cache['head']
     for layer in layers:
      for head in heads:
        pd.set_option("display.max_rows", None, "display.max_columns", None)
        print(load_matrix(out_dir,name,str(layer),str(head)))
        if len(layers) > 2 or len(heads) > 2:
         res = input(MSG_PRESS_CMD)
         if res in ['exit','stop','bye']:
           break
                
  elif cmds[0] == 'load':
    print(LOAD_WARN)
  
  elif cmds[0] == 'good':
   try:
     parse_select(['select',cmds[1]],0)    
   except Exception:
     print('> Error')
   gh()
  
  elif cmds[0] == 'help':
   help()
     
  elif cmds[0] == 'see':
      try:
       token = cmds[1]
       add_element_cache('token',token)
      except Exception:
       console_show(MSG_ERROR_NO_TOKEN)
       break 
      while cache_contain('name') == False:
       name = input(INPUT_NAME)
       add_element_cache('name',name)
         
     
  elif cmds[0] in ['exit','stop','bye']:
      print('bye')
      sys.exit()   
   
  elif cmds[0] == 'name':
   try:
    cache['name'] = cmds[1]
   except Exception:
    continue
       
  else:  
      try: 
        parse_select(['select',cmds[0]],0)
        see()
        sme()         
          
      except Exception:
        console_show('Error') 
      
                     
 sys.exit()

