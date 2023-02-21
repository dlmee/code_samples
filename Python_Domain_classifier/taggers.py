import numpy as np
from scipy.sparse import vstack
from model.SKLModel import *
from math import log


class Tagger:

    def __init__(self, model: Built_Model, targets: list):
        self.targets = self.preptarget(targets)
        self.model = model
        self.transition_probs = self.find_transition_prob(model.master)

    def preptarget(self, targets):
        prepped = []
        for sentence in targets:
            sent = []
            for word in sentence:
                pair = (word, 'UNK')
                sent.append(pair)
            if len(sent) != 0:
                prepped.append(sent)
        return prepped

    def createfeatures(self, master, tpurpose):  # In the train function. need tpurpose to be True
        features = []
        target = []
        word = []

        for sentence in master:

            for i, pair in enumerate(sentence):
                feat = {}
                feat["token"] = pair[0]
                feat["length"] = len(pair[0])

                if i > 0:
                    feat["pos-1"] = sentence[i - 1][1]
                    feat["pos-2"] = '<s>'
                    if re.match('[A-Z]', pair[0][0]):
                        feat["cap"] = "True"
                    else:
                        feat["cap"] = "False"
                    if i > 1:
                        feat["pos-2"] = sentence[i - 2][1]

                if i == 0 or tpurpose == True:  # test and dev == True. Erasing to make it POS 'blind'
                    feat["pos-1"] = '<s>'
                    feat["pos-2"] = '<s>'
                    feat["cap"] = "False"

                if len(pair[0]) > 2:
                    feat["pre2"] = pair[0][:2]
                    feat["pre3"] = pair[0][:3]
                    feat["suf2"] = pair[0][-2:]
                    feat["suf3"] = pair[0][-3:]

                features.append(feat)
                target.append(pair[1])
                word.append(pair[0])

        return features, target, word

    def inverse_label_index(self, numlabel, model):
        allclasses = list(model.label_encoder.classes_)
        return allclasses[int(numlabel)]

    def greedy(self):
        targets = self.targets
        model = self.model
        # Using the given model.
        allpredictions = []

        # Step 1, use the createfeatures function to mirror data structure.
        if len(targets) > 1:
            devsen = []
            for sentence in targets:
                sentence = [sentence]
                devs, target, word = self.createfeatures(sentence, True)
                devsen.append(devs)
        else:
            devsen, target, word = self.createfeatures(targets, True)
            devsen = [devsen]
        # Step 2 use logistic regression

        for sentence in devsen:
            predictions = []
            for i, pred in enumerate(sentence):
                if i > 0:
                    posneg = {}
                    posneg["pos-1"] = predictions[i - 1]
                    pred.update(posneg)
                    if i > 1:
                        posneg = {}
                        posneg["pos-2"] = predictions[i - 2]
                        pred.update(posneg)
                transformed = model.feature_encoder.transform(pred)  # Transform 1 at a time.

                if i == 0:
                    pfsparse = transformed
                else:
                    pfsparse = vstack([pfsparse, transformed]).toarray()

                prediction = model.model.predict(transformed)
                predictions.append(self.inverse_label_index(prediction, model))
            allpredictions.append(predictions)

        #print(allpredictions)
        return allpredictions


    def find_transition_prob(self, master):

        self.transitions = ['<s>']
        for sentence in master:
            for pair in sentence:
                self.transitions.append(pair[1])
            self.transitions.append('<s>')

        transition_probs = self.create_transition_table(self.transitions)

        return transition_probs

    def inverse_log_prob_index(self, logndarray):
        allclasses = list(range(len(logndarray[0])))
        logprobs = logndarray[0]
        assert len(allclasses) == len(logprobs)

        matched = []
        for i, pos in enumerate(allclasses):
            matched.append((logprobs[i], pos))
        top3 = [(-50, "POS"), (-50, "POS"), (-50, "POS")]
        for match in matched:
            for i, val in enumerate(top3):
                if match[0] > val[0]:
                    top3[i] = match
                    break
        # print(top3)
        top3prob = []
        top3pos = []
        for value in top3:
            top3prob.append(value[0])
            top3pos.append(value[1])
        return top3prob, top3pos  # Just need to decide how I want the return. Do I want the value and the POS, or just the POS?

    def sentence_chunker(self, sentence, maxsize):
        if len(sentence) < maxsize:
            return [sentence]
        for i in range(2, 100):
            if len(sentence) / i < maxsize:
                chunks = i
                chsentences = []
                for k in range(chunks):
                    schunk = []
                    for j in range(maxsize):
                        if sentence != []:
                            schunk.append(sentence.pop(0))
                        else:
                            break
                    chsentences.append(schunk)
                return chsentences

    def stacker(self, probstack, posstack, top3prob, top3pos):
        current = np.shape(probstack[0])  # always two values such as 9,2 or 27,3 or 81,4
        if current == (3,):
            c1 = 3
            c2 = 1
        else:
            c1 = current[0]
            c2 = current[1]
        nprobstack = np.zeros((c1 * 3, c2 + 1))
        nposstack = np.zeros((c1 * 3, c2 + 1))
        counter = -1
        ncounter = -1
        for i, ppset in enumerate(probstack):  # probstack is always going to be three
            nprob = top3prob[i]
            for pastprobs in ppset:
                counter += 1
                nprobstack[counter] = np.hstack((nprob, pastprobs))
        for i, ppset in enumerate(posstack):  # posstack is always going to be three
            npos = top3pos[i]
            for pastpos in ppset:
                ncounter += 1
                nposstack[ncounter] = np.hstack((npos, pastpos))
        return nprobstack, nposstack

    def create_transition_table(self, tags):
        tprob = {}
        tbase = {}
        for i, tag in enumerate(tags):
            if tag in tbase:
                tbase[tag] += 1
            else:
                tbase[tag] = 1
            if i == 0:
                continue
            tseq = tags[i - 1] + '|' + tag
            if tseq in tprob:
                tprob[tseq] += 1
            else:
                tprob[tseq] = 1
        for k, v in tprob.items():
            base = re.search('[^|]+', k)
            tprob[k] = v / tbase[base.group()]
        return tprob

    def viterbi_init(self, model):  # -> Tuple[NDArray, NDArray, TagSeq]
        targets = self.targets
        self.predictions = []
        self.predictions_sentence = []
        for sentence in targets:
            single_sentence = []
            # print(sentence)
            sench = self.sentence_chunker(sentence, 5)  # max size basically enforcing beam search
            # print(sench, "this is the list of chunks")
            for i, chunk in enumerate(sench):
                # print("This is the chunk", chunk)
                if chunk == []:
                    # print("EMPTY CHUNK!")
                    break
                devsen, target, word = self.createfeatures([chunk],
                                                           True)  # Have to send as a list, so that it is compatible.

                # This happens between feature creation and TRANSFORMATION
                if i > 0:  # basically a second or later chunk.
                    # print(devsen, self.finalpredictions, "The problem is here")
                    devsen[0]['pos-1'] = self.predictions[-1]
                    devsen[0]['pos-2'] = self.predictions[-2]
                    # Need to override the POS-1 and 2 features before predicting probability.
                # print("\ndevsen0\n", devsen[0] )
                featuredword = model.feature_encoder.transform(
                    devsen[0])  # Transform just first word, the rest happens recursively.

                logprobs = model.model.predict_log_proba(featuredword)
                top3prob, top3pos = self.inverse_log_prob_index(logprobs)

                if len(chunk) != 1:
                    fprobstack = []
                    fposstack = []
                    for possiblePOS in top3pos:
                        # print(devsen[1:], 'going into the recursive world')
                        probs, posmatrix = self.recursive_viterbi(self.inverse_label_index(possiblePOS, model), "<s>",
                                                                  devsen[1:], model)
                        fprobstack.append(probs)
                        fposstack.append(posmatrix)
                    finalprobs, finalpos = self.stacker(fprobstack, fposstack, top3prob, top3pos)
                    # print("we made it!")
                    best = []
                    transcost = []
                    for j, row in enumerate(finalpos):
                        # Each row is going to be a sequence of tags, revealed numerically.
                        rowtvalue = 0
                        for m, pos in enumerate(row):
                            if m == 0:
                                continue
                            key = self.inverse_label_index(row[m - 1], model) + '|' + self.inverse_label_index(pos,
                                                                                                               model)
                            if key not in self.transition_probs:
                                self.transition_probs[key] = .00001
                            tc = self.transition_probs[key]
                            rowtvalue += (log(tc) / 4)
                        transcost.append(rowtvalue)
                    # print('transcost list', transcost)

                    for k, row in enumerate(finalprobs):
                        value = np.sum(row, dtype='float32')
                        value += transcost[k]
                        best.append((value, k))
                    best = sorted(best, key=lambda x: x[0], reverse=True)
                    f = list(finalpos[best[0][1]])
                    # print("What I'm looking for")
                    for p in f:
                        self.predictions.append(self.inverse_label_index(p, model))
                        single_sentence.append(self.inverse_label_index(p, model))
                else:
                    # a chunk with the length of 1.
                    self.predictions.append(self.inverse_label_index(top3pos[0], model))
                    single_sentence.append(self.inverse_label_index(top3pos[0], model))

                # print("END OF A CHUNK", self.finalpredictions)
                # After a chunk has been processed, now we need to finalize those POS.
            # print("END OF A SENTENCE", self.finalpredictions)
            self.predictions_sentence.append(single_sentence)
        # print("The very end", self.finalpredictions)
        return self.predictions

    def recursive_viterbi(self, pos_1, pos_2, devsen, model):  # Each pass the sentence will get smaller and smaller.
        # override the POS on the already featured sentence.
        devsen[0]['pos-1'] = pos_1
        devsen[0]['pos-2'] = pos_2
        # Transform the word.
        featuredword = model.feature_encoder.transform(devsen[0])
        # Do the prediction
        logprobs = model.model.predict_log_proba(featuredword)
        top3prob, top3pos = self.inverse_log_prob_index(logprobs)
        if len(devsen) == 1:
            return top3prob, top3pos
        else:
            probstack = []
            posstack = []
            # Need to change back to POS
            for i, possiblePOS in enumerate(top3pos):
                p1 = self.inverse_label_index(possiblePOS, model)
                p2 = pos_1
                probs, pospeeches = self.recursive_viterbi(p1, p2, devsen[1:], model)
                probstack.append(probs)
                posstack.append(pospeeches)

            nprobstack, nposstack = self.stacker(probstack, posstack, top3prob, top3pos)

            return nprobstack, nposstack