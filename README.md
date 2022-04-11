# liBERTy
This project is compatible only with 12 x 12 (layers x heads) BERT model. 
## Setup
Create the enviornment
```
cd liBERTy

conda env create -f environment.yml
conda activate liberty
```
Set config file (liBERTy/config.txt)
```
model_dir = /path/model/dir  
out_dir = /path/output/dir  
name = default_name_of_sentence
```
## Load a sentence

Given a sentence you need to create a txt file:
```
[sentence]  

sentence = text of the sentence

sentence_id = id used for it
```
then use this command 

```
cd liBERTy

python liberty load -f /path/to/file_txt -n name
```
where name is the name of the sentence given by user


## Commands

We can find two different python Facade scripts. The first *liBERTy* is used to load sentence and analyze, the second onw *talking_heads* for visualizing.

### stat

```
python liberty.py stat -t [token] -n [name of the sentence] -v [entropy/noop/me]
```
Given a token visualize values of a metric (entropy,noop,me) for each heads in the model

If -t = all comute avg and std

### see

for use see command exec *talking_heads* first

```
python talking_heads.py
```
select token to visualize

```
see [token]
```

select the head
```
[layer],[head]
```
For example

```
see Hello
6,7
```

## good

Given an head visualize for each token in the sentence if there's high value of noop

```
python talking_heads.py

good [layer],[head]
```
