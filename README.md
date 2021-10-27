# liBERTy

Comandi principali per questa versione (in futuro forse modificata per renderla più intuitiva):

*L'opzione -n è sempre presente ma per alleggerire la scrittura è stato omesso (di default se omesso si considera il nome presente in liBERTy/config.txt*

## Preparazione ambiente
Basta utilizzare il file requirment per caricare l'ambiente conda liberty

## Definizione del file di config

in liBERTy/config.txt si trovano tre opzioni importanti da indicare:

[config]  
model_dir = /path/model/dir  
out_dir = /path/output/dir  
name = default_name_of_sentence


## Caricamento frase

Prima di essere utilizzata una frase deve essere caricata attraverso il comando:
python liberty.py load -f file/path -n name

-f indica il file .txt contenente la frase e l'id della frase seguendo questa struttura:

[sentence]  

sentence = ...

sentence_id = ...

il nome -n sarà l'id a cui sarà associato nella directory di output

## Esempi

### see

* *Data una frase e un token vedere il livello di attention data da ogni token della frase (printa tutti i possibili token presenti nella frase che combaciano con il token indicato nell'opzione -w):*  

python liberty.py see -l 3 -h 5 -w lesioni -n 1

### stat 

Dato un token e una combinazione delle possibili opzioni printa:

* *Box plot di tutte le head del valore JSD dato un vettore (entropy o noop)*
  
python liberty.py stat -t all -v entropy

* *Dato un token il valore JSD in ogni head num = indice che indica quale token selezionare di quelli identici presenti nella frase (vedi comando find)*  

python liberty.py stat -t lesioni {-i num} {-v entropy}

* *Data una percentuale trova la frequenza di tutte le head presenti in quel range di JSD dato un vettore (entropy o noop): il range di jsd consiedrato è p < x < p+10 dove p è il valore indicato (in percentuale) nell'opzione -p*   

python liberty.py stat -p 50 {-v entropy}
 
* *Dato un token printa sul piano cartesiano la posizione delle head dati i due vettori (entropy e noop)*  

python liberty stat -t lesioni -v all
 
### outlier 

* *Data una head (l,h) trova per un vettore specificato (entropy o noop) i primi 10 token pù divergenti dalla media JSD:*  

python liberty.py outlier -l 4 -h 6 -v entropy

* *Dato un vettore ordina le head in base alla media delle differenze dei primi 10 token pù divergenti dalla media JSD della head stessa:*  

python liberty.py outlier -l all -v entropy

### who

* *Data una head (l,h) e una percentuale p trova tutte i token che hanno nella head indicata un valore di JSD x compreso tra p < x < p+10 (funziona solo per il vettore entropy per ora)*  

python liberty.py who -l 4 -h 6 -p 80

### smear 

* *Data una head (l,h) e un token della frase trova lo smear complessivo (tutte i token che hanno un valore di attention accettabile intorno al token indicato) (non ancora fixato il problema degli indici quindi è utilizzabile solo per i token che appaiono nella frase una sola volta), smear indica come "smear" quelli vicini mentre i "drops" sono quelli non smear quindi distaccati dalla sbavatura*  

python liberty.py smear -l 8 -h 8 -t lesioni

### find 
* *Dato un token vede se e in che quantità è presente nella frase, in caso sia presente più volte indica anche i possibili indici di ogni token, gli indici servono a certi comandi per individuare quale token tra quelli presenti utilizzare*  

python liberty.py find -t lesioni

