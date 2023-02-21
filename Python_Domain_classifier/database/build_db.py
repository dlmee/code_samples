from database.mysqlconnector import *
import time
from pathlib import Path
import re
import csv

class DBuilder:
    def __init__(self, lexicon, vectricon, breakdown, morphicon, inflections, documents):
        self.repo = MysqlRepository()
        ct = time.perf_counter()
        self.db_lexicon(lexicon, breakdown)
        nt = time.perf_counter()
        print('lexicon populated', nt - ct)
        self.db_vectricon(vectricon)
        nt = time.perf_counter()
        print('vectricon populated', nt - ct)
        self.db_morphicon(morphicon)
        nt = time.perf_counter()
        print('morphicon populated', nt - ct)
        self.db_inflecticon(inflections)
        nt = time.perf_counter()
        print('inflecticon populated', nt - ct)
        self.db_sentences(documents)
        nt = time.perf_counter()
        print('sentences populated', nt - ct)


    # lexicon with morphemic breakdown, indexed sentences
    # vectricon columns: words, vectors (word, num)
    # morphicon: morphemes, list of words with that morpheme
    # inflecticon: word, list of words that are inflected from that word.
    # sentences:

    def db_lexicon(self, lexicon: dict, breakdown: dict):
        uploader = []
        for k, v in lexicon.items():
            morphemes = breakdown[k]
            sm = '-'.join(morphemes)
            uploader.append((k[:30], sm[:30], str(v)[:800]))
        self.repo.cursor.executemany("INSERT INTO lexicon VALUES (0, %s, %s, %s)", uploader)

    def db_vectricon(self, vectricon: dict):
        uploader = []
        for key, dval in vectricon.items():
            sub = ''
            for j, r in dval.items():
                newstring = str(j) + '-' + str(r) + ','
                sub = sub + newstring
            uploader.append((key, sub[:800]))
        self.repo.cursor.executemany("INSERT INTO vectricon VALUES (0, %s, %s)", uploader)

    def db_morphicon(self, morphicon):
        uploader = []
        for k, v in morphicon.items():
            newv = ', '.join(v)
            uploader.append((k, newv[:500]))
        self.repo.cursor.executemany("INSERT INTO morphicon VALUES (0, %s, %s)", uploader)


    def db_inflecticon(self, inflections: dict):
        uploader = []
        for k, v in inflections.items():
            if len(v) == 1:
                uploader.append((k, v[0]))
            else:
                uploader.append((k, ', '.join(v)[:300]))
        self.repo.cursor.executemany("INSERT INTO inflecticon VALUES (0, %s, %s)", uploader)

    def db_sentences(self, documents):
        uploader = []
        for i, doc in enumerate(documents):
            uploader.append((i+1, doc))
        self.repo.cursor.executemany("INSERT INTO sentences VALUES (%s, %s)", uploader)




class POSbuilder:
    def __init__(self):
        self.repo = MysqlRepository()
        #pos = self.load_csv_pos() #This means that it will load from the CSV and then populate the database

    def db_pos(self, pos):
        uploader = []
        for k, v in pos.items():
            uploader.append((k, v[0][0] + ':' + str(v[0][1]), v[1][0] + ':' + str(v[1][1]), v[2][0] + ':' + str(v[2][1]), v[3][0] + ':' + str(v[3][1]), v[4][0] + ':' + str(v[4][1])))
        self.repo.cursor.executemany("INSERT INTO partsofspeech VALUES (0, %s, %s, %s, %s, %s, %s)", uploader)

    def load_csv_pos(self):
        topfive = {}
        file = Path("""/home/dlmee/ILAD_Internship/domain_classifier/model/corpus/id_pos.csv""")
        with open(file, newline='') as csvfile:
            ddata = csv.reader(csvfile, delimiter=',')
            topfive = {}
            subps = {'ADJ': 0, 'ADP': 0, 'ADB': 0, 'ADV': 0, 'AUX': 0, 'CCONJ': 0, 'DET': 0, 'INTJ': 0, 'NOUN': 0, 'NUM': 0,
                     'PART': 0, 'PRON': 0, 'PROPN': 0, 'PUNCT': 0, 'SCONJ': 0, 'SYM': 0, 'VERB': 0, 'X': 0, '_': 0}
            for row in ddata:
                topfive[row[1]] = {}
                for counts in row[2:]:
                    pos, count = re.split(':', counts)
                    topfive[row[1]][pos] = count
        formatted = {}
        for k, v in topfive.items():
            pairs = list(v.items())
            formatted[k] = pairs
        return formatted

        #pbuilder = build_db.POSbuilder(topfive)

class domaintable:
    def __init__(self, domaindata): #domaindata should be of the class domain

        #Prep both tables

        self.dprepped = []
        self.dtarget = []
        self.traverse(domaindata, [])


        #Now need to build a dynamic table code for the domaintable

        self.dynamic = ["word VARCHAR(50)"]
        self.d2 = ["%s"]
        for i in reversed(range(domaindata.level)):
            self.dynamic.append("level{} VARCHAR(10)".format(i))
            self.d2.append("%s")
        self.dynamic.append("PRIMARY KEY (word)")
        self.dynamic = ",".join(self.dynamic)
        self.d2 = ",".join(self.d2)
        self.basesqlbuild = ["USE ikatadomain", "DROP TABLE IF EXISTS domainicon", "CREATE TABLE domainicon(Q)"]
        self.basesqlbuild[-1] = re.sub("Q", self.dynamic, self.basesqlbuild[-1])
        self.exmanydynamic = "INSERT INTO domainicon VALUES (Q)"
        self.exmanydynamic = re.sub("Q", self.d2, self.exmanydynamic)
        self.repo = MysqlRepository()

        for command in self.basesqlbuild:
            self.repo.cursor.execute(command)

        self.repo.cursor.executemany(self.exmanydynamic, self.dprepped)

        #Build the keystones table

        self.dynamic = ["domain VARCHAR(50)", "level INT"]
        self.d2 = ["%s", "%s"]
        for k in domaindata.keystones.keys():
            self.dynamic.append("{} FLOAT(10)".format(k))
            self.d2.append("%s")
        self.dynamic.append("PRIMARY KEY (domain)")
        self.dynamic = ",".join(self.dynamic)
        self.d2 = ",".join(self.d2)
        self.basesqlbuild = ["USE ikatadomain", "DROP TABLE IF EXISTS keystones", "CREATE TABLE keystones(Q)"]
        self.basesqlbuild[-1] = re.sub("Q", self.dynamic, self.basesqlbuild[-1])
        self.exmanydynamic = "INSERT INTO keystones VALUES (Q)"
        self.exmanydynamic = re.sub("Q", self.d2, self.exmanydynamic)

        for command in self.basesqlbuild:
            self.repo.cursor.execute(command)

        self.repo.cursor.executemany(self.exmanydynamic, self.dtarget)

    def traverse(self, domaindata, traversal):
        counter = 0
        for k, v in domaindata.children.items():
            counter += 1

            t2 = traversal + [str(counter)]
            t2 = ".".join(t2)
            temp = [t2, v.level]
            for k2, v2 in v.keystones.items():
                temp.append(v2)#The key comes back at the build level.
            self.dtarget.append(temp)

            if v.level == 0: #This is the bottom of the recursion
                for word in v.vector:
                    compound = []
                    for i, level in enumerate(traversal):
                        if i == 0: compound.append(level)
                        else: compound.append(".".join([compound[-1], level]))
                    compound.append(".".join([compound[-1], str(counter)]))
                    self.dprepped.append([word] + compound)
            else:
                traversal.append(str(counter))
                self.traverse(v, traversal) #recursion
                discard = traversal.pop()


class DEmpty:
    def __init__(self):
        self.repo = MysqlRepository()
        self.emptydb()
    def emptydb(self):
        self.repo.cursor.execute("DELETE FROM lexicon")
        self.repo.cursor.execute("DELETE FROM vectricon")
        self.repo.cursor.execute("DELETE FROM morphicon")
        self.repo.cursor.execute("DELETE FROM inflecticon")
        self.repo.cursor.execute("DELETE FROM sentences")
        self.repo.cursor.execute("DELETE FROM partsofspeech")


if __name__ == "__main__":
    pass