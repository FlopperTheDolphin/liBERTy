from fun.vs_constants import SEPARATOR
def help(command):
 if command == 'load':
  return "load [-f | --file = <path to file sentence>] [-n | --name = <name of sentence>]"
 elif command == 'see':
  return "see [-l | --layer = <layer>] [-h | --head = <head>] [-w | --word = <word>] [-n | --name] [-t | --time = <time>] - show most weighted token given a word: sentence = change style of visualisation, time = select how many tokens correlated find"
 elif command == 'stat':
  return "stat [-t | --token = <token to visualise>] [-n | --name = <name of the sentence>] [-p | --perc = <div value>] [-v | --vector = <vector used for shannon>] [-i | --id = <number of token>] - print all the Shannon distance from a spaecified token for all the matrix from a specified vector, if there's perc print all the Shannon distance from a specified token from matrix with value x in range --perc < x < --perc+10, you can also define which vectore use among: entropy and noop. If token == 'all' plot statistic values"
 elif command == 'who':
  return "who [-l | --layer = <layer>] [-h | --head = <head>] [-n | --name = <name of sentence>] [-p | --perc = <div value>] - find matrix which has div value x such that: --perc < x < --perc+10, for a specified token"
 elif command == 'cluster':
  return "cluster [-r | --centroids = <number of centroids>]- cluster matrix Mx leads metric |M1-M2|"
 elif command == 'pos':
  return "pos [-l | --layer = <layer>] [-h | --head = <head>] - show the pos graph given a attention matrix"
 elif command == 'smear':
  return "smear [-l | --layer = <layer>] [-h | --head = <head>] [-n | --name = <name of sentence>] [-t | --token = <token>] - given a token spot the smear"
 elif command == 'tokens':
  return "tokens - print all sentence tokens"   
 elif command == 'sentence':
  return "sentence - print the sentence"   
 elif command == 'find':
  return "find [-t | --token] [-n | --name] - underline all token in the sentence"
   
def total_help():
 print('* ' + help('load'))
 print()

 print('* ' + help('see'))
 print()
 print('* ' + help('smear'))
 print()

 print('* ' + help('stat'))
 print()
 print('* ' + help('who'))

 print()
 print('* ' + help('cluster'))

 print()
 print('* ' + help('pos'))   

 print()
 print('* ' + help('tokens'))
 print()
 print('* ' + help('sentence'))
 print()
 print('* ' + help('find'))   
