import spacy
from gensim.models import Word2Vec
from spacy.symbols import LEMMA, ORTH, POS


attention = True
nlp = spacy.load('it')
adenopatie = [{ORTH: 'adenopatie', LEMMA: 'adenopatia', POS: 'NOUN'}]
invariato = [{ORTH: 'invariato', LEMMA: 'invariare', POS: 'VERB'}]
invariata = [{ORTH: 'invariata', LEMMA: 'invariare', POS: 'VERB'}]
invariati = [{ORTH: 'invariati', LEMMA: 'invariare', POS: 'VERB'}]
invariate = [{ORTH: 'invariate', LEMMA: 'invariare', POS: 'VERB'}]
nlp.tokenizer.add_special_case(u"adenopatie", adenopatie)
nlp.tokenizer.add_special_case('invariato', invariato)
nlp.tokenizer.add_special_case('invariata', invariata)
nlp.tokenizer.add_special_case('invariate', invariate)
nlp.tokenizer.add_special_case('invariati', invariati)
immodificato = [{ORTH: 'immodificato', LEMMA: 'invariare', POS: 'VERB'}]
immodificata = [{ORTH: 'immodificata', LEMMA: 'invariare', POS: 'VERB'}]
immodificati = [{ORTH: 'immodificati', LEMMA: 'invariare', POS: 'VERB'}]
immodificate = [{ORTH: 'immodificate', LEMMA: 'invariare', POS: 'VERB'}]
nlp.tokenizer.add_special_case(u"adenopatie", adenopatie)
nlp.tokenizer.add_special_case('immodificato', immodificato)
nlp.tokenizer.add_special_case('immodificata', immodificata)
nlp.tokenizer.add_special_case('immodificate', immodificate)
nlp.tokenizer.add_special_case('immodificati', immodificati)
addensamenti = [{ORTH: 'addensamenti', LEMMA: 'addensamento', POS: 'NOUN'}]
nlp.tokenizer.add_special_case('addensamenti', addensamenti)
concepts = ['nodulo', 'lesione', 'adenopatia', 'addensamento', 'margine', 'alterazione', 'ispessimento',
            'linfonodo', 'placca', 'formazione', 'versamento', 'tessuto', 'massa', 'nodulare', 'enfisema', 'pareti']
CONCEPTS_LEMMAS = list()
for c in concepts:
    lemma = nlp(c)[0].lemma_
    CONCEPTS_LEMMAS.append(lemma)
LIVELLO_1 = 'TIPO_ESAME'
LIVELLO_2 = 'RISULTATO_ESAME'
LIVELLO_3 = 'NATURA_LESIONE'
LIVELLO_4 = 'SITO_LESIONE'
LIVELLO_5 = 'TIPO_LESIONE'
PRIMO_ESAME = 'PRIMO ESAME'
FOLLOW_UP = 'FOLLOW-UP'
NEGATIVO = 'NEGATIVO'
POSITIVO = 'POSITIVO'
STABILE = 'STABILE'
PROGRESSIONE_RECIDIVA = 'PROG. RECIDIVA'
NON_NEOPLASTICO = 'NON NEOPLASTICO'
NATURA_DUBBIA = 'NATURA DUBBIA'
NEOPLASTICO = 'NEOPLASTICO'
POSITIVO_NON_NEOPLASTICO = 'POSITIVO NON NEOPLASTICO'
NON_POSITIVO_NON_NEOPLASTICO = 'NON POSITIVO NON NEOPLASTICO'
DUBBIO_O_NEOPLASTICO = 'NEO O DUBBIO'
FALSO = 'FALSO'
VERO = 'VERO'
level_dictionary = {1: LIVELLO_1, 2: LIVELLO_2, 3: LIVELLO_3, 4: LIVELLO_4, 5: LIVELLO_5}
livello_1 = {PRIMO_ESAME: 0, FOLLOW_UP: 1}
livello_2_binario = {NEGATIVO: 0, POSITIVO: 1}
livello_2_primo_esame = {NEGATIVO: 0, POSITIVO: 1}
livello_2_follow_up = {NEGATIVO: 0, STABILE: 1, PROGRESSIONE_RECIDIVA: 2}
livello_2_completo = {NEGATIVO: 0, POSITIVO: 1, STABILE: 2, PROGRESSIONE_RECIDIVA: 3}
stabile_recidiva = {STABILE: 0, PROGRESSIONE_RECIDIVA: 1}
livello_positivo_non_neoplastico = {POSITIVO_NON_NEOPLASTICO: 0, NON_POSITIVO_NON_NEOPLASTICO: 1}
livello_3 = {NON_NEOPLASTICO: 0, NATURA_DUBBIA: 1, NEOPLASTICO: 2}
livello_3_accorpato = {NON_NEOPLASTICO: 0, DUBBIO_O_NEOPLASTICO: 1}
livello_4 = {FALSO: 0, VERO: 1}
TOTAL_WORDS = 450
word2vec = Word2Vec.load('embedding/referti_totale_word_embedding_200.model')
tag2vec = Word2Vec.load('embedding/referti_totale_pos_embedding_10.model')
VECTOR_SIZE = word2vec.vector_size
POS_VECTOR_SIZE = tag2vec.vector_size
PRIMO_LIVELLO = 'TIPO_ESAME'
SECONDO_COMPLETO = 'RISULTATO_COMPLETO'
POSITIVE_NON_NEO = 'POSITIVE_NON_NEO'
SECONDO_PE = 'RISULTATO_PRIMO_ESAME'
SECONDO_FW = 'RISULTATO_FOLLOW_UP'
SECONDO_BINARIO = 'RISULTATO_POSITIVO_NEGATIVO'
STABILE_PROGRESSIONE_RECIDIVA = 'STABILE_RECIDIVA'
TERZO_POSITIVE = 'NATURA_LESIONE'
TERZO_POSITIVE_NO_DUBBIA = 'NATURA_LESIONE_NO_DUBBIA'
TERZO_COMPLETO = 'NATURA_LESIONE_COMPLETO'
TERZO_ACCORPATO = 'SOSPETTO'
QUARTO_LIVELLO = 'SITO_LESIONE'
POLMONE = 'POLMONE'
PLEURA = 'PLEURA'
MEDIASTINO = 'MEDIASTINO'
tutti_i_livelli = [PRIMO_LIVELLO, SECONDO_COMPLETO, SECONDO_BINARIO, TERZO_COMPLETO, TERZO_ACCORPATO]
BASE_MODEL = 'word_model'
TAG_MODEL = 'word_tag_model'
LABEL_MODEL = 'word_label_model'
TAG_LABEL_MODEL = 'word_tag_label_model'
SENTENCE_BLOCK_MODEL = 'sentence_block_model'
ATTENTION_MODEL = 'attention_model'