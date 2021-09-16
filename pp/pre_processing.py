from bs4 import BeautifulSoup
from os import listdir
from data_model import Report, Classification, Request, NERToken
from constants import *
from spacy.tokens import Doc
from typing import List
import pickle
from input_processing import ReportInstance, Dataset, DividedReportInstance
import numpy as np


def extract_text_from_reports(xml_file):
    """
    Extracts report id, user, a spacy Doc (text) containing and the classification from the XML file.
    Retokenizes the report header (TC TORACE etc.) into a single token in order to allow the division in sentences.
    """
    contents = xml_file.read()
    soup = BeautifulSoup(contents, "xml")
    referto = soup.find('REFERTO')
    id = soup.find('IDENTIFICATIVO_REFERTO').get_text()
    text = referto.get_text()
    text = text.replace('-', ' ')
    utente = referto.get('utente')
    classificazione = soup.find('CLASSIFICAZIONE')
    list_values = [(s, classificazione.find(s).get_text()) for s in level_dictionary.values()]
    classificazione = Classification(list_values)
    doc = nlp(text)
    tokens = list(doc)
    caps_lock = True
    i = 0
    while caps_lock and i < len(tokens):
        token_text = tokens[i].text
        token_tag = tokens[i].pos_
        if not token_text.isupper() and token_tag != 'PUNCT' and token_tag != 'SPACE':
            caps_lock = False
        else:
            i += 1
    intestazione = doc[0:i]
    with doc.retokenize() as retokenizer:
        retokenizer.merge(intestazione)
    return id, utente, doc, classificazione


def text_filtering(doc: Doc, annotated: bool = False):
    """
    Extracts only the TORACE part of the report, excluding all the other sections.
    Keeps also the summary part (labeled usually as CONCLUSIONI).
    """
    sents = list(doc.sents)
    cleaned_sents = list()
    for s in sents:
        if len(s.text) >= 2:
            cleaned_sents.append(s)
    # print(sents)
    selected = list()
    section_titles = 'COLLO', 'ENCEFALO', 'ADDOME', 'MASSICCIO FACCIALE', 'AORTA', 'ANGIO', 'Addome', 'CUORE', 'Encefalo', 'ENCEFALO'
    torace = True
    for i in range(0, len(cleaned_sents)):
        text = cleaned_sents[i].text
        for s_t in section_titles:
            if s_t in text and ('MDC' not in text or 'TC' not in text):
                torace = False
        if 'TORACE' in text or 'Torace' in text or 'polmone' in text:
            torace = True
        if 'CONCLUSIONI' in text or 'Conclusioni' in text:
            torace = True
        if torace:
            selected.append(cleaned_sents[i])
    new_text = ''
    for span in selected:
        span_text = span.text.lower().replace('ml.', 'ml')
        span_text = span_text.lower().replace('e.v.', 'ev')
        span_text = span_text.lower().replace('m.d.c.', 'mdc')
        span_text = span_text.lower().replace('cm.', 'cm')
        span_text = span_text.replace('(', '')
        span_text = span_text.replace(')', '')
        if not annotated:
            if span[0].text.lower() == 'torace':
                span_text = span_text.lower().replace('torace ', '')
                span_text = span_text.lower().replace('torace:', '')
        new_text = new_text + span_text + ' '
    new_doc = nlp(new_text)
    return new_doc


def create_reports_from_folder(folder_path: str) -> List[Report]:
    """
    Creates a list of Report objects from a folder containing XML files
    """
    files = listdir(folder_path)
    report_list = list()
    for j in range(len(files)):
        f = files[j]
        infile = open(folder_path+"/"+f, "r", encoding='utf-8')
        id, utente, doc, classificazione = extract_text_from_reports(infile)
        new_doc = text_filtering(doc)
        report = Report(id, utente, doc, new_doc, classificazione)
        print(id)
        report_list.append(report)
    return report_list


def create_all_reports_dump():
    """
    Creates a pickle dump of all the radiology reports.
    """
    reports = create_reports_from_folder('referti_2019/xml')
    dump_file = open('reports_repository.pkl', 'wb')
    pickle.dump(reports, dump_file)


def get_ids_controllati() -> List[str]:
    """
    Get ids from radiology reports which were controlled
    :return: list of ids (strings)
    """
    path = 'referti_2019/controllati'
    files = listdir(path)
    ids = list()
    for f in files:
        xml_file = open(path+'/'+f, 'r')
        contents = xml_file.read()
        soup = BeautifulSoup(contents, "xml")
        id = soup.find('IDENTIFICATIVO_REFERTO').get_text()
        ids.append(id)
    return ids


def input_preparation_test():
    folder_path = 'referti_2019/xml'
    files = listdir(folder_path)
    for j in range(len(files)):
        f = files[j]
        xml_file = open(folder_path+"/"+f, "r", encoding='utf-8')
        id, utente, doc, classificazione = extract_text_from_reports(xml_file)
        new_doc = text_filtering(doc)
        report = Report(id, utente, doc, new_doc, classificazione)
        print(new_doc)
        instance = ReportInstance(report)
        doc_input = instance.word_input
        matrix = np.array(doc_input)
        assert matrix.shape == (TOTAL_WORDS, VECTOR_SIZE)


def divisione_frasi(report: Report) -> dict:
    new_doc = report.filtered_doc
    first_sentences = list()
    positive_sentences = list()
    negative_sentences = list()
    invariate_sentences = list()
    sents = list(new_doc.sents)
    conclusioni = False
    final_sentences = list()
    for i in range(len(sents)):
        s = sents[i]
        root = s.root
        children = list(root.children)
        tags = [t.tag_ for t in children]
        word_lemmas = [t.lemma_ for t in s]
        # print(word_lemmas)
        if not conclusioni:
            if 'conclusione' in word_lemmas:
                conclusioni = True
            if 'invariare' in word_lemmas:
                invariate_sentences.append(s)
            elif 'BN__PronType=Neg' in tags and 'non' in word_lemmas:
                negative_sentences.append(s)
            elif i < 3 and ('esame' in word_lemmas or 'tc' in word_lemmas or 'confrontare' in word_lemmas):
                found = False
                for lemma in word_lemmas:
                    if lemma in CONCEPTS_LEMMAS:
                        found = True
                        positive_sentences.append(s)
                    if lemma == 'mese' or lemma == 'anno':
                        found = True
                        final_sentences.append(s)
                if not found:
                    first_sentences.append(s)
            else:
                positive_sentences.append(s)
        else:
            final_sentences.append(s)
    sentence_blocks = dict()
    sentence_blocks.__setitem__('first_sentences', first_sentences)
    sentence_blocks.__setitem__('positive_sentences', positive_sentences)
    sentence_blocks.__setitem__('negative_sentences', negative_sentences)
    sentence_blocks.__setitem__('invariate_sentences', invariate_sentences)
    sentence_blocks.__setitem__('final_sentences', invariate_sentences)
    return sentence_blocks


def sentence_filtering(reports: List[Report]) -> List[Report]:
    new_reports = list()
    for report in reports:
        new_text = ''
        new_doc = report.filtered_doc
        positive_sentences = list()
        negative_sentences = list()
        sents = list(new_doc.sents)
        for i in range(len(sents)):
            s = sents[i]
            root = s.root
            children = list(root.children)
            tags = [t.tag_ for t in children]
            word_lemmas = [t.lemma_ for t in s]
            # print(word_lemmas)
            if 'BN__PronType=Neg' in tags and 'non' in word_lemmas:
                negative_sentences.append(s)
            else:
                positive_sentences.append(s)
        for span in positive_sentences:
            span_text = span.text
            new_text = new_text + span_text + ' '
        new_doc = nlp(new_text)
        report.filtered_doc = new_doc
        new_reports.append(report)
    return new_reports


def negative_reports_filtering(folder_path: str) -> List[Report]:
    files = listdir(folder_path)
    report_list = list()
    only_negatives = 0
    con_conclusioni = 0
    scarti = 0
    for j in range(len(files)):
        f = files[j]
        infile = open(folder_path+"/"+f, "r", encoding='utf-8')
        id, utente, doc, classificazione = extract_text_from_reports(infile)
        new_doc = text_filtering(doc)
        report = Report(id, utente, doc, new_doc, classificazione)
        print(id)
        first_sentences, positive_sentences, negative_sentences, invariate_sentences, final_sentences = divisione_frasi(report)
        # print(first_sentences)
        # print(positive_sentences)
        # print(negative_sentences)
        # print(invariate_sentences)
        # print(final_sentences)
        if len(first_sentences) + len(positive_sentences) + len(negative_sentences) + len(invariate_sentences) + len(final_sentences) == 0:
            scarti += 1
        elif len(positive_sentences) + len(invariate_sentences) == 0:
            only_negatives += 1
            print(first_sentences, positive_sentences, negative_sentences, invariate_sentences)
        if len(final_sentences) > 0:
            con_conclusioni += 1
        report_list.append(report)
    print(only_negatives)
    print(con_conclusioni)
    return report_list


def extract_text_from_request(request_xml) -> Request:
    # The request is already BeautifulSoup
    #contents = request_xml.read()
    #soup = BeautifulSoup(contents, "xml")
    soup = request_xml # TODO: replace this line
    text = soup.find('TESTO').get_text()
    text = text.replace('\n', ' ')
    id_tag = soup.find('ID_CLASSIFICAZIONE')
    if id_tag is not None:
        id = id_tag.get_text()
    else:
        id = ''
    doc = nlp(text)
    filtered_doc = text_filtering(doc)
    request = Request(id, doc, filtered_doc)
    return request


def create_divided_reports_from_folder(folder_path: str):
    files = listdir(folder_path)
    for j in range(len(files)):
        f = files[j]
        infile = open(folder_path+"/"+f, "r", encoding='utf-8')
        id, utente, doc, classificazione = extract_text_from_reports(infile)
        new_doc = text_filtering(doc)
        report = Report(id, utente, doc, new_doc, classificazione)
        sentence_blocks = divisione_frasi(report)
        print(sentence_blocks)


def parse_classification_line(classification_line: str) -> Classification:
    classification_line = classification_line.replace(' ', '')
    levels = classification_line.split(';')
    assert len(levels) >= 4
    value_list = list()
    uno = due = tre = 'nonset'
    if levels[0] == 'fwup':
        uno = FOLLOW_UP
    if levels[0] == 'esame1':
        uno = PRIMO_ESAME
    if levels[1] == 'pos':
        due = POSITIVO
    if levels[1] == 'neg':
        due = NEGATIVO
    if levels[1] == 'progrecid':
        due = PROGRESSIONE_RECIDIVA
    if levels[1] == 'stabile':
        due = STABILE
    if levels[2] == '-':
        tre = ''
    if levels[2] == 'neopl':
        tre = NEOPLASTICO
    if levels[2] == 'non-neopl':
        tre = NON_NEOPLASTICO
    if levels[2] == 'lesionedub':
        tre = NATURA_DUBBIA
    if 'nonset' in uno+due+tre:
        return None
    value_list.append((level_dictionary.get(1), uno))
    value_list.append((level_dictionary.get(2), due))
    value_list.append((level_dictionary.get(3), tre))
    classification = Classification(value_list)
    return classification


def create_annotated_reports(folder_path: str) -> List[Report]:
    reports = list()
    infixes = ['(', ')', '/', '-', ';', '*']
    files = listdir(folder_path)
    splits_dict = dict()
    classification_dict = dict()
    for j in range(len(files)):
        f = open(folder_path+'/'+files[j], 'r', encoding='utf-8')
        id = files[j].split('-')[0].replace('.txt', '')
        lines = f.readlines()
        analysis = lines[0]
        classification_line = analysis.split(':')[1]
        classification = parse_classification_line(classification_line)
        if classification is None:
            continue
        classification_dict.__setitem__(id, classification)
        splits = [s.replace('\n', '').split('\t') for s in lines if not s.startswith('#')]
        referto_index = -1
        referto = ['Ref', 'REFERTO', 'Referto', 'ref', 'REF']
        for k in range(len(splits)):
            if splits[k][0] in referto:
                referto_index = k
                break
        torace = ['TORACE', 'Torace', 'Esame', 'Indagine']
        torace_index = -1
        if referto_index == -1:
            for k in range(len(splits)):
                if splits[k][0] in torace:
                    torace_index = k
                    break
        if referto_index != -1:
            start_index = referto_index+1
        elif torace_index != -1:
            start_index = torace_index
        else:
            start_index = 0
        ner_tokens = [NERToken(s[0], s[1]) for s in splits[start_index:] if len(s) >= 2 and s[0] != ':']
        if id not in splits_dict.keys():
            splits_dict.__setitem__(id, [ner_tokens])
        else:
            splits_list = splits_dict.get(id)
            splits_list.append(ner_tokens)
            splits_dict.__setitem__(id, splits_list)
    for id in splits_dict.keys():
        final_tokens = list()
        splits_list = splits_dict.get(id)
        if len(splits_list) == 1:
            final_tokens = splits_list[0]
        else:
            merged_list = dict()
            first_tokens = splits_list[0]
            words = [t.word for t in first_tokens]
            for i in range(len(words)):
                for j in range(len(splits_list)):
                    if len(splits_list[j]) != len(words):
                        print(id)
                    word = words[i]
                    label = splits_list[j][i].label
                    if i not in merged_list.keys():
                        merged_list.__setitem__(i, NERToken(word, label))
                    else:
                        ner_token = merged_list.get(i)
                        # print(word, ner_token.label, label)
                        if ner_token.label == 'O' and len(label) > len(ner_token.label):
                            new_token = NERToken(word, label)
                            merged_list.__setitem__(i, new_token)
            for i in merged_list.keys():
                final_tokens.append(merged_list.get(i))
        spaces = list()
        for i in range(len(final_tokens)-1):
            actual = final_tokens[i]
            next = final_tokens[i+1]
            if actual.word in infixes or next.word in infixes:
                space = False
            else:
                space = True
            spaces.append(space)
        spaces.append(False)
        words = [t.word for t in final_tokens]
        try:
            doc = Doc(nlp.vocab, words, spaces)
        except ValueError:
            doc = Doc(nlp.vocab, words)
        for name, proc in nlp.pipeline:
            doc = proc(doc)
        # sents = [s for s in doc.sents]
        # print(sents)
        classification = classification_dict.get(id)
        report = Report(id, 'annotated_report', original_doc=doc, filtered_doc=text_filtering(doc, annotated=True),
                        classification=classification)
        report.set_annotated_tokens(final_tokens)
        reports.append(report)
    return reports


# reports = create_annotated_reports('referti_2019/referti_iob2')
# print(len(reports))
# negative_reports_filtering('referti_2019/xml')
# create_divided_reports_from_folder('referti_2019/xml')