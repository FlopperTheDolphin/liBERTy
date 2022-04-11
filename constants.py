

TKN_LOADING = 'loading Tokenizer...'
TKN_LOADED = 'Tokenizer loading completed'
MSG_MDL_LOADING= 'loading Model...'
MSG_MDL_LOADED = 'Model loading completed'
SNT_LOADED = 'Sentece loading completed'
MSG_MTX_CAL = 'Model comp attentions...'
MSG_MTX_COMP = 'Attentions calculus completed'
MSG_MTX_SAVE = 'Save all the attentions matrix...'
MSG_MTX_SAVE_COMP = 'All attentions matrix are saved'
VW_20_TOKENS = '---------------20 Parole con più alto valore associato a '
VW_20_TOKENS2 = '---------------'
MSG_TOKENS_COMP = 'Comp bert tokens and spacy tokens...'
MSG_TOKEN_COMP_END = 'Comp tokens completed'
MSG_UNIFY_TOKEN = 'Unify bert ancd spacy tokens...'
MSG_UNIFY_TOKEN_END = 'Unification completed' 
MSG_MTX_DIR = 'Directory generated at:'
MSG_MAX_MTX='max attention value for the matrix:'
UNDERLINE='UNDERLINE'
MSG_AN_TOKEN='= token chosen'
MSG_YELLOW='YELLOW'
MSG_RED='RED'
MSG_GREEN='GREEN'
CHOSEN_WORD = '[underline]'
MSG_TOP_TOKENS = 'tokens with higher attentions values: '
MSG_SENT='Sentence:'
MSG_COMP_COLUMN='selection of all the columns...'
MSG_COMP_COLUMN_END='end of selection'
TOKEN ='TOKEN:'
MSG_HEAD_CHOSEN='Head chosen:'
MSG_FIRST_MTX = 'First 10 matrix with deivergent higher divergent index:'
MSG_HEAD_SORTED='Heads sorted by frequency:'
MSG_TOKEN_SUS='Token suspect:'
MSG_HEADS_SUS='Heads suspect:'
MSG_FIRST_DIV_MTX='First 10 matrix with higher divergence for:'
MSG_FILE_NOT_FOUND='File Not found, exec before command stat'
MSG_CHOSEN_TOKEN_GIVEN_HEAD = 'Tokens with perc selected in head:'
MSG_GRAPH_SAVED='Graph saved at:'
SEPARATOR ='-------------------------------------------------------------------'
MSG_WARNING_TIME = 'This could take more times...'
MSG_TOTAL_STAT = 'Comp all stat for all tokens'
PERC= 'Perc'
MSG_COMP_AVG = 'Comp stat' 
MSG_MAX_AVG = 'Max AVG value'
MSG_MAX_STD = 'Max STD value'
MSG_ORDERD_FOR_ATT = 'Token ordered by sum of attention value:' 
MSG_ORDER_ATT_SUM = 'Head sorted by max attention sum'
MSG_POSSIBLE_NOOP='Possible noop:'
MSG_NO_NOOP = 'No noop:'
MSG_MAX_VALUE = 'Max attention values possible:'
MSG_TOKEN_ID='Token id: '
MSG_COMPARE='Compare two different metrics'
MSG_DONE_1='First Metric comp done'
MSG_DONE_2='Second Metric comp done'
JSD_COMP = 'JSD comparison for token:'
MSG_POSSIBLE_INDEX='Possible id for token:'
MSG_NO_INDEX="There's only one token"
ID_TOKEN='id token:'
MSG_EXEC_STAT='exec before command stat -t all -v '
MSG_MAX_OUTLIER = 'Tokens with max JSD difference:'
MSG_NO_TOKEN = 'No token in the sentence'
MSG_PRESS_CMD='liBERTy> '
MSG_ERROR_TMA='too much arguments'
INPUT = 'press any key to continue... '
MSG_ERROR_SYNTAX_1 = 'ERROR wrong syntax ---> num1:num2,num3:num4'
MSG_ERROR_SYNTAX_2 = 'ERROR wrong syntax: num1:num2 ---> num1 < num2'
MSG_ERROR_SYNTAX_3 = 'ERROR wrong syntax: num1 is not a number'
MSG_ERROR_NO_TOKEN = 'ERROR wrong syntax: see token'
ALL_LAYERS = 'select all layers'
TALKING_HEADS = 'TALKING HEADS'
MAG_NOT_LOAD = 'exec command python lib -f file/to/sentence -n name, this console not consider the loading command'
INPUT_NAME = 'Specify sentence... '
INPUT_HEAD= 'select an head... '
SCORE_COM = 'all score are saved...'
MSG_SORTED = 'All outlier are sorted' 
TOTAL= 'total:'
COMMAND = 'COMMANDS:'
SEE = 'see [token to see]: select the token'
LH = '[layer,head]: if a token is selected see attention distribution for self attention [layer,head]'
GOOD = 'good [layer,head]: compute noop values for each token and find good ones'
NAME='name [name sentence]: change sentence to see'
BE='bye/exit: to exit'
LOAD_WARN='exec command python lib -f file/to/sentence -n name, this console not consider the loading command'
#Colors
GREEN = 'rgb(0,255,0)'
YELLOW = 'rgb(255,255,0)'
RED = 'rgb(255,0,0)'

def chosen_token(token):
 return 'Token: ' + str(token)

def msg_red(time,token):
 return str('= first '+ str(time)+' token with highest attention value for ' + token)

def msg_yellow(time,token):
 return str('= from '+ str(int(time+1)) +' to '+str(int(time+time))+' tokens for ' + token)
 
def msg_green(time,token):
 return msg_yellow(time+time,token) 
 
def higher_token(time,token):
 return str('first ' + str(time) + ' token with higher attention values for ' + token)

 
