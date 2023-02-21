import re
from database.mysqlconnector import MysqlRepository


class Dloader:
    def __init__(self):
        self.repo = MysqlRepository()

    def loadall(self, source):
        tables = {'lexicon': self.lexicon, 'vectricon': self.vectricon, 'morphicon': self.morphicon,
                  'inflecticon': self.inflecticon, 'sentences': self.sentences, 'partsofspeech': self.pos}
        query = f'SELECT * FROM {source};'
        self.repo.cursor.execute(query)
        target = list(self.repo.cursor)
        download = tables[source](target)
        return download

    def domains(self, target):
        query = "DESCRIBE keystones;"
        self.repo.cursor.execute(query)
        dt = list(self.repo.cursor)
        totals = {}
        mapping = {}
        for i, t in enumerate(dt[2:]):
            totals[t[0]] = []
            mapping[i] = t[0]
        target = target.lower()
        target = target.split()
        for t in target:
            query = f'SELECT * FROM domainicon WHERE word = "{t}";'
            self.repo.cursor.execute(query)
            levels = list(self.repo.cursor)
            if not levels: continue
            levels = list(levels[0])
            levels.reverse()
            for l in levels[:-1]:
                query = f'SELECT * FROM keystones WHERE domain = "{l}"'
                self.repo.cursor.execute(query)
                domains = list(self.repo.cursor)
                domains = domains[0]
                for i, d in enumerate(domains[2:]):
                    totals[mapping[i]].append(d)
        for k, v in totals.items():
            totals[k] = sum(v)
        return totals

    def lexicon(self, download):
        lex = {}
        breakdown = {}
        for line in download:
            bdown = re.findall('\w+', line[2])
            indices = re.findall('\w+', line[3])
            lex[line[1]] = indices
            breakdown[line[1]] = bdown
        return (lex, breakdown)


    def vectricon(self, download):
        vectricon = {}
        for vector in download:
            separated = re.split(',|-', vector[2])
            reconstituted = {}
            for i, value in enumerate(separated):
                if i % 2 == 1:
                    if len(value) != 1:
                        value = 1
                    reconstituted[separated[i - 1]] = int(value)
            vectricon[vector[1]] = reconstituted
        return vectricon

    def morphicon(self, download):
        morph = {}
        for line in download:
            indices = re.findall('\w+', line[2])
            morph[line[1]] = indices
        return morph

    def inflecticon(self, download):
        inflecticon = {}
        for line in download:
            indices = re.findall('\w+', line[2])
            inflecticon[line[1]] = indices
        return inflecticon

    def sentences(self, download):
        return [sentence[1] for sentence in download]

    def pos(self, download):
        pos = {}
        for line in download:
            pos[line[1]] = [line[2], line[3], line[4], line[5], line[6]]
        return pos