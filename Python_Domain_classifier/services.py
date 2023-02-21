import database.load_db
import model.inflections
from database import build_db
from database.mysqlconnector import *
from model import vectors
from model import morph
from model import docs
from app.taggers import *
from s_comparison.primer import *
from s_comparison.verifier import *
from joblib import load
from domains import domain_distributor
import time

class Services:

    def __init__(self):
        #self.repo = MysqlRepository()
        pass

    def build_model(self):
        ct = time.perf_counter()
        self.model = load('model1.joblib')
        # self.model = SKLModel.Built_Model()
        self.tagger = Tagger
        nt = time.perf_counter()
        print('tagger', nt - ct)
        lexicon = docs.Documents()
        self.lexicon = lexicon.lexicon
        self.documents = lexicon.documents
        nt = time.perf_counter()
        print('lexicon, sentences', nt - ct)
        vectricon = vectors.Vectricon(lexicon.documents)
        self.vectricon = vectricon.mv
        nt = time.perf_counter()
        print('vectricon', nt - ct)
        self.sw = stopwords.words('indonesian')
        morphemes = morph.Morphemes(self.documents)
        self.breakdown = morphemes.all_true_m
        self.morphicon = morphemes.rev_morph_index
        nt = time.perf_counter()
        print('morphicon', nt - ct)
        inflect = model.inflections.Inflections(self.lexicon, self.breakdown, self.vectricon, self.morphicon)
        self.inflecticon = inflect.inflections
        nt = time.perf_counter()
        print('inflecticon', nt - ct)
    def sentence_comparison(self, sentences: list) -> float:
        assert len(sentences) == 2
        instance = Primer(sentences[0], sentences[1], self.lexicon, self.vectricon, self.sw, self.model, self.tagger, self.breakdown, self.morphicon, self.inflecticon)
        oracle = Verifier(instance.sen1primed, instance.sen2primed)
        return oracle.finalprob

    def find_pos(self):
        POSdict = {}
        subps = {'ADJ': 0, 'ADP': 0, 'ADB': 0, 'ADV': 0, 'AUX': 0, 'CCONJ': 0, 'DET': 0, 'INTJ': 0, 'NOUN': 0, 'NUM': 0,
                 'PART': 0, 'PRON': 0, 'PROPN': 0, 'PUNCT': 0, 'SCONJ': 0, 'SYM': 0, 'VERB': 0, 'X': 0, '_': 0}
        print(len(self.sentences))
        for i, sentence in enumerate(self.sentences):
            if i % 100 == 0:
                print(i)
            processed = Sentence(sentence, self.sw, self.model, self.tagger)
            for pair in processed.combined:
                if pair[0] in POSdict:
                    POSdict[pair[0]][pair[1]] += 1
                else:
                    POSdict[pair[0]] = subps.copy()
                    POSdict[pair[0]][pair[1]] += 1
        topfive = {}
        for k, v in POSdict.items():
            pairs = list(v.items())
            pairs.sort(key=lambda x: x[1], reverse=True)
            topfive[k] = pairs[:5]
        pbuilder = build_db.POSbuilder() #doesn't need return value, currently topfive is
        pbuilder.db_pos(topfive) #This will build after a run.
        return topfive



    def build_db(self):
        builder = build_db.DBuilder(self.lexicon, self.vectricon, self.breakdown, self.morphicon, self.inflecticon, self.documents)

    def empty_db(self):
        destructor = build_db.DEmpty()

    def build_domains(self, target):
        layer = domain_distributor.Domainicon() #I don't think these are necessary in this call: target, self.vectricon, self.lexicon, self.sentences
        sv = layer.reshape(target)
        print(len(sv))
        refined = []
        for pair in sv:
            try:
                if self.pos[pair[0]][0][:4] == 'NOUN':
                    refined.append(pair)
            except:
                pass
        print(len(refined))
        final = layer.builddomains(refined, self.vectricon)
        dbf = build_db.domaintable(final)
        return final
    def load_all(self):
        ct = time.perf_counter()
        downloader = database.load_db.Dloader()
        lexicon = downloader.loadall('lexicon')
        self.lexicon = lexicon[0]
        self.breakdown = lexicon[1]
        self.morphicon = downloader.loadall('morphicon')
        self.vectricon = downloader.loadall('vectricon')
        self.inflecticon = downloader.loadall('inflecticon')
        self.sentences = downloader.loadall('sentences')
        self.sw = stopwords.words('indonesian')
        self.model = load('model1.joblib')
        self.tagger = Tagger
        self.pos = downloader.loadall('partsofspeech')
        nt = time.perf_counter()
        print('Download entire database', nt - ct)

    def use_domainicon(self, target):
        downloader = database.load_db.Dloader()
        domains = downloader.domains(target) #target should be tokenized sentence
        return domains

    def test_doc(self, targetdirectory):
        testsentences = open(targetdirectory)
        testlines = testsentences.readlines()
        BItl = []
        ENtl = []
        for i, line in enumerate(testlines):
            if i % 2 == 0:
                BItl.append(line[:-1])
            else:
                ENtl.append(line[:-1])
        return BItl, ENtl



if __name__ == "__main__":
    #Build the service layer first
    service = Services()

    #########
    #If you need to build the database

    #service.empty_db()
    #service.build_model()
    #service.build_db()
    #loader = build_db.POSbuilder() #variable is the class,
    #fromcsv = loader.load
    #fromcsv = loader.load_csv_pos() #Unless you want to do it the long way, then call: service.find_pos() Then don't run the line below
    #loader.db_pos(fromcsv)
    #########

    #If you need to load the database

    service.load_all()

    ########
    #If you want to build the domainicon,you must have all the variables that you would get from the above service.load_all()

    service.build_domains(service.vectricon)

    ########
    #If you want to use the domain classifier, you do not have to use service.load_all(), but you must have a populated database from service.build_domains()

    #answer = service.use_domainicon("Kita harus mengingat dan menuliskan masa lalu agar tidak menjadi tragedi")
    #print(answer)

    #########

    # If you want to use the sentence comparison

    # answer = service.sentence_comparison(['pria itu duduk di kursi hijau', 'wanita itu meninggalkan mobil biru'])
    # print(answer)
