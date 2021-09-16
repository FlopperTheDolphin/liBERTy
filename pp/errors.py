class SecondLevelError(Exception):

    def __init__(self, val: int):
        self.value = val

    def __str__(self):
        if self.value == 0:
            return 'Il referto è un primo esame, chiamare secondo_livello_primo_esame'
        else:
            return 'Il referto è un follow-up, chiamare secondo_livello_follow_up'


class NegativeError(Exception):

    def __init__(self, level: int):
        self.level = level

    def __str__(self):
        if self.level == 2:
            return 'Il referto è negativo, per cui non può essere nè stabile, nè progressione recidiva'
        if self.level == 3:
            return 'Il referto è negativo, per cui non ha natura lesione specificata'
        if self.level == 4:
            return 'Il referto è negativo, per cui non ha sito specificato'
        if self.level == 5:
            return 'Il referto è negativo, per cui non ha tipo lesione specificato'


class TypeMissingError(Exception):

    def __str__(self):
        return 'Il tipo di vettore può essere solamente word (200) o pos (10)'