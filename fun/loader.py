import sys
import os
import json
# import xml.etree.ElementTree as ET
# import requests
import pandas as pd
from os import listdir
from fun.vs_constants import *
from transformers import AutoModel, AutoTokenizer, BertConfig
import numpy as np
from fun.view import console_show
from bs4 import BeautifulSoup


def get_ref_from_xml(ref_xml_path, name_file):
    files = listdir(ref_xml_path)
    for f in files:
        if (str(f) == str(name_file)):
            xml_file = open(ref_xml_path + '/' + f, 'r')
            contents = xml_file.read()
            soup = BeautifulSoup(contents, "xml")
            break
    return soup.find('REFERTO').get_text()


def load_model(bert_path, tok_path):
    config = BertConfig.from_pretrained(bert_path, output_hidden_states=True, output_attentions=True)
    tokenizer = load_tokenizer(bert_path)
    model = AutoModel.from_pretrained(bert_path, config=config)

    return tokenizer, model


def save_matrix(dir_path, tokens, attentions, verbose=True):
    # dimension (12,1,12,n,n)

    dic_max = dict()
    for i in range(12):
        for j in range(12):
            np_attention = attentions[i][0][j].detach().numpy()
            df = pd.DataFrame(data=np_attention, columns=tokens)
            file_name = dir_path + "/" + "att-mtx_layer-" + str(i + 1) + "_head-" + str(j + 1) + ".csv"
            df.to_csv(file_name, index=False)
            dic_max[str((i + 1, j + 1))] = df.max().max() + 0
            if (verbose):
                console_show('[' + file_name + '] saved')

    save_in_json(dic_max, dir_path + "/max.json")


# def load_sentece(sent_xml_path,name_file):
# sentence = get_ref_from_xml(sent_xml_path,name_file)       
# print('> '+ SNT_LOADED)
# return sentence

# def load_sentence(name_file):
# return "TC TORACE-ADDOME Esame condotto prima e dopo somministrazione di mdc endovena (Xenetix 350, 120 ml), confrontato con precedente del 12/11/2018. TORACE: rimossi la cannula tracheale ed il drenaggio toracico destro. Non significative modificazioni della componente di entità non elevata di emotorace destro, moderato incremento della componente liquida di versamento associata. Persiste addensamento compressivo del parenchima adiacente. Lieve incremento del versamento pleurico sinistro, sempre di entità non elevata, ed associato a fenomeni di disventilazione compressiva del parenchima adiacente. Non sanguinamenti in atto. Nettamente ridotto lo scollamento pleurico destro, permane minimo in sede basale anteriore. Invariato il resto. ADDOME: non modificazioni significative degli ematomi in sede epatica ed al polo renale superiore destro. Non sanguinamenti in atto. Lieve incremento dell'entità del versamento liquido periepatico, lungo la doccia parietocolica destra, nello scavo pelvico ed in sede perisplenica (invariata la minima componente ematica in fase di organizzazione nel contesto del versamento pelvico). Non sovradistensione delle anse intestinali, che hanno pareti di normali spessore ed enhancement. Invariato il resto.",'prova'

def load_matrix(out_dir, id_sent, layer, head):
    file_path = out_dir + "/" + id_sent + "/" + "att-mtx_layer-" + layer + "_head-" + head + ".csv"

    return pd.read_csv(file_path)


def load_tokenizer(bert_path):
    return AutoTokenizer.from_pretrained(bert_path)


def load_from_json(path_file):
    f = open(path_file, 'r')
    data = json.load(f)
    f.close()
    return data


def save_in_json(name, path_file):
    out_file = open(path_file, "w")
    json.dump(name, out_file, indent=6)
    out_file.close()


def save_sentence(sentence, sent_id, sent_path):
    dic_sent = dict()
    dic_sent['sentence'] = sentence
    dic_sent['sent_id'] = sent_id
    save_in_json(dic_sent, sent_path)
