from errors import SecondLevelError, NegativeError
from spacy.tokens import Doc
from typing import List
from constants import *


class NERToken:
    def __init__(self, word: str, label: str):
        self.word = word
        self.label = label

    def __str__(self):
        return str(self.word) + ' ' + str(self.label_terzo())

    def __len__(self):
        return len(self.word)

    def label_primo(self):
        if 'B-TES' in self.label:
            return 'TES'
        if 'I-TES' in self.label:
            return 'TES'
        return 'O'

    def label_terzo(self):
        if 'B-NEO' in self.label or 'B-LES' in self.label in self.label:
            return 'NEO'
        if 'I-NEO' in self.label or 'I-LES' in self.label in self.label:
            return 'NEO'
        return 'O'

    def label_sito(self):
        if 'B-SIT' in self.label:
            return 'SIT'
        if 'I-SIT' in self.label:
            return 'SIT'
        return 'O'

    def label_totale(self):
        if self.label != 'O' and 'ESA' not in self.label and self.label != 'B-SIT' and self.label != 'I-SIT':
            return 'REL'
        return 'O'


class Request:
    def __init__(self, id: str, original_doc: Doc, filtered_doc: Doc):
        self.id = id
        self.original_doc = original_doc
        self.filtered_doc = filtered_doc


class Classification:
    """
    Class modeling the hierarchical classification
    LEVEL 1: TIPO ESAME (PRIMO ESAME/FOLLOW-UP)
    LEVEL 2: RISULTATO ESAME (NEGATIVO/POSITIVO - NEGATIVO/STABILE/PROGRESSIONE RECIDIVA)
    LEVEL 3: NATURA LESIONE (NON NEOPLASTICO/NATURA DUBBIA/NEOPLASTICO)
    LEVEL 4: SITO LESIONE
    LEVEL 5: TIPO LESIONE
    """

    def __init__(self, list_values: List):
        self.tipo_esame = ''
        self.risultato_esame = ''
        self.natura_lesione = ''
        self.sito_lesione = ''
        self.tipo_lesione = ''
        for (level, value) in list_values:
            if level == level_dictionary.get(1):
                self.tipo_esame = value
            if level == level_dictionary.get(2):
                self.risultato_esame = value
            if level == level_dictionary.get(3):
                self.natura_lesione = value
            if level == level_dictionary.get(4):
                self.sito_lesione = value
            if level == level_dictionary.get(5):
                self.tipo_lesione = value


class Report:
    """
    Class modeling the radiology report
    :param id: report identifier
    :param user: user identified
    :param original_doc: spacy Doc containing the original text
    :param filtered_doc: spacy Doc containing the filtered text
    :param classification: Classification object containing the actual class values given by the doctors
    """

    def __init__(self, id: str, user: str, original_doc: Doc, filtered_doc: Doc, classification: Classification):
        self.id = id
        self.user = user
        self.original_doc = original_doc
        self.filtered_doc = filtered_doc
        self.classification = classification
        self.annotated_tokens: List[NERToken] = None

    def set_annotated_tokens(self, tokens: List[NERToken]):
        self.annotated_tokens = tokens

    def annotated_indexes(self):
        return [i for i in range(len(self.annotated_tokens)) if self.annotated_tokens[i].label_totale() != 'O']


    """
    Methods retrieving the class value for each layer according to the dictionaries in constants.py
    Ex. Primo livello -> {0, 1} with 0 = PRIMO ESAME, 1 = FOLLOW-UP 
    """
    def primo_livello(self) -> int:
        return livello_1.get(self.classification.tipo_esame)

    def secondo_livello_primo_esame(self) -> int:
        if self.primo_livello() == livello_1.get(PRIMO_ESAME):
            return livello_2_primo_esame.get(self.classification.risultato_esame)
        else:
            raise SecondLevelError(self.primo_livello())

    def secondo_livello_follow_up(self) -> int:
        if self.primo_livello() == livello_1.get(FOLLOW_UP):
            return livello_2_follow_up.get(self.classification.risultato_esame)
        else:
            raise SecondLevelError(self.primo_livello())

    def secondo_livello_completo(self) -> int:
        return livello_2_completo.get(self.classification.risultato_esame)

    def secondo_livello_binario(self) -> int:
        secondo_completo = self.secondo_livello_completo()
        if secondo_completo == livello_2_completo.get(POSITIVO):
            return livello_2_binario.get(POSITIVO)
        if secondo_completo == livello_2_completo.get(STABILE):
            return livello_2_binario.get(POSITIVO)
        if secondo_completo == livello_2_completo.get(PROGRESSIONE_RECIDIVA):
            return livello_2_binario.get(POSITIVO)
        if secondo_completo == livello_2_binario.get(NEGATIVO):
            return livello_2_binario.get(NEGATIVO)

    def positivo_non_neoplastico(self) -> int:
        if self.primo_livello() == livello_1.get(FOLLOW_UP):
            return livello_positivo_non_neoplastico.get(NON_POSITIVO_NON_NEOPLASTICO)
        else:
            if self.secondo_livello_completo() == livello_2_completo.get(POSITIVO) and self.terzo_livello() == livello_3.get(NON_NEOPLASTICO):
                return livello_positivo_non_neoplastico.get(POSITIVO_NON_NEOPLASTICO)
            else:
                return livello_positivo_non_neoplastico.get(NON_POSITIVO_NON_NEOPLASTICO)

    def terzo_livello(self) -> int:
        if self.secondo_livello_completo() == livello_2_completo.get(NEGATIVO):
            raise NegativeError(3)
        else:
            return livello_3.get(self.classification.natura_lesione)

    def terzo_livello_con_negative(self) -> int:
        if self.secondo_livello_completo() == livello_2_completo.get(NEGATIVO):
            return livello_3.get(NON_NEOPLASTICO)
        else:
            return livello_3.get(self.classification.natura_lesione)

    def terzo_livello_accorpato(self) -> int:
        if self.terzo_livello_con_negative() != livello_3.get(NON_NEOPLASTICO):
            return livello_3_accorpato.get(DUBBIO_O_NEOPLASTICO)
        else:
            return livello_3_accorpato.get(NON_NEOPLASTICO)

    def terzo_livello_positive_accorpato(self) -> int:
        if self.secondo_livello_completo() == livello_2_completo.get(NEGATIVO):
            raise NegativeError(3)
        else:
            if self.terzo_livello() != livello_3.get(NON_NEOPLASTICO):
                return livello_3_accorpato.get(DUBBIO_O_NEOPLASTICO)
            else:
                return livello_3_accorpato.get(NON_NEOPLASTICO)

    def stabile_recidiva(self) -> int:
        if self.primo_livello() == livello_1.get(FOLLOW_UP):
            if self.secondo_livello_completo() == livello_2_completo.get(NEGATIVO):
                raise NegativeError(2)
            else:
                return stabile_recidiva.get(self.classification.risultato_esame)
        else:
            raise SecondLevelError(self.primo_livello())

    def quarto_livello(self):
        print(self.classification.sito_lesione)
        polmone = pleura = mediastino = 0
        if 'POLMONE' in self.classification.sito_lesione:
            polmone = 1
        if 'PLEURA' in self.classification.sito_lesione:
            pleura = 1
        if 'MEDIASTINO' in self.classification.sito_lesione:
            mediastino = 1
        return [polmone, pleura, mediastino]

    def polmone(self):
        polmone = 0
        if 'POLMONE' in self.classification.sito_lesione:
            polmone = 1
        return polmone

    def pleura(self):
        pleura = 0
        if 'PLEURA' in self.classification.sito_lesione:
            pleura = 1
        return pleura

    def mediastino(self):
        mediastino = 0
        if 'MEDIASTINO' in self.classification.sito_lesione:
            mediastino = 1
        return mediastino

    def label_per_livello(self, livello: str):
        """
        Retrieves the classification label given the level
        :param livello: the level string (see constants.py)
        """
        if livello == PRIMO_LIVELLO:
            return self.primo_livello()
        if livello == SECONDO_COMPLETO:
            return self.secondo_livello_completo()
        if livello == SECONDO_PE:
            return self.secondo_livello_primo_esame()
        if livello == SECONDO_FW:
            return self.secondo_livello_follow_up()
        if livello == SECONDO_BINARIO:
            return self.secondo_livello_binario()
        if livello == TERZO_POSITIVE:
            return self.terzo_livello()
        if livello == TERZO_COMPLETO:
            return self.terzo_livello_con_negative()
        if livello == POSITIVE_NON_NEO:
            return self.positivo_non_neoplastico()
        if livello == TERZO_ACCORPATO:
            return self.terzo_livello_accorpato()
        if livello == STABILE_PROGRESSIONE_RECIDIVA:
            return self.stabile_recidiva()
        if livello == QUARTO_LIVELLO:
            return self.quarto_livello()
        if livello == POLMONE:
            return self.polmone()
        if livello == PLEURA:
            return self.pleura()
        if livello == MEDIASTINO:
            return self.mediastino()