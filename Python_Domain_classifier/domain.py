import numpy as np

class Domain:

    def __init__(self, level):
        self.name = None
        self.identity = False
        self.integrity = 0
        self.level = level
        self.parent = None
        self.children = {}
        self.age = 0
        self.vector = []
        self.dcontext = None
        self.dtargets = [("pemerintah", ["pemerintah", "walikota", "kantor", "presiden", "raja", "kekuasaan", "terpilih", "hukum", "istana", "suara", "pemimpin", "pajak", "negara"]), ("sejarah", ["sejarah", "masa lalu", "cerita", "peristiwa", "ingat", "ceritakan kembali", "tulis", "penting", "tragedi", "rakyat", "akun", "dokumen", "budaya ", "agama", "hari", "perubahan"])]
        self.keystones = {}



    def grow(self, target):
        current = self.level
        descent = [self]
        while current > 1:
            descent = list(descent[0].children.values()) 
            descent.sort(key = lambda x:len(x.children.keys()))
            current = descent[0].level
        if current == 1:
            keys = len(descent[0].children.keys())
            child = Domain(current - 1)
            child.vector = target
            descent[0].children[str(keys + 1)] = child
        else:
            print("WARNING: This is a level 0 domain node")

    def adopt(self, name, child):
        self.children[name] = child
    
    def shrink(self, key):
        return self.children.pop(key)

    def keystone(self):
        if self.children:
            child = list(self.children.keys())[0]
            self.dtargets = self.children[child].dtargets
        for target in self.dtargets:
            self.keystones[target[0]] = 0
        for line in self.dtargets:
            for word in line[1]:
                if word in self.vector: self.keystones[line[0]] += 1
        for k, v in self.keystones.items():
            percentage = np.log(v/len(self.vector) + 1)
            if len(self.vector) == 0: print(self.name)
            self.keystones[k] = percentage
