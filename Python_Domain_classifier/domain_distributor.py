from typing import Dict
import numpy as np
from model.domain import Domain
import random
import time
import re

# type aliases
Target = Dict[str, Dict[str, dict]]
Grouping = Dict[str, list]
Supervector = Dict[str, dict]


class Domainicon:

    def __init__(self):  # target equals a supervector, context equals all vectors.
        self.dtotals = []
        self.totprob = []
        self.rejects = {}
        self.lastchance = {}
        self.gcounter = 0

    def builddomains(self, nouns, context):
        starter = nouns[:5000] # Just for experimentation
        oldslice = 0
        newslice = 100
        postcount = 6
        domain_tol = -18
        recyclecount = 10
        siftcount = 4
        droptolerance = 2  # dtol * siftcount = total amount of words dropped.
        domainlevel = 0
        gmetric = 25
        layercap = 12

        # Build posts

        heads = []

        # Grow posts
        while newslice < len(starter):
            newbatch = starter[oldslice:newslice]
            print(len(newbatch), "length of newbatch", postcount, "And post count")
            ct = time.perf_counter()

            #############

            if domainlevel == 1:
                recycled = self.recycle(self.rejects, recyclecount)
                if recycled: Head.grow(recycled)
                Head.age += 1
                Head = self.growpost(Head, newbatch, postcount, context, droptolerance)
                Head, totalintegrity, domaincount = self.integrity(Head, context)
                difference = totalintegrity + abs(domain_tol)

            ##############

            if domainlevel == 0:
                Head = self.findposts(newbatch, postcount, context, droptolerance, siftcount)
                Head, totalintegrity, domaincount = self.integrity(Head, context)
                domainlevel += 1
                difference = 0

            ##############

            if postcount > layercap:
                print("forming domain")
                Head = self.package(Head, context)
                heads.append(Head)
                domainlevel = 0
                gmetric = 100
                postcount = 6
                difference = 0
                if siftcount < 21: siftcount += 1
                domain_tol -= 8

            if difference < 0:
                difference = int(abs(difference // siftcount))
                for i in range(0, difference):
                    Head.grow([])
                postcount += difference




            print('words processed in this run', len(newbatch), "total words processed: ", newslice)
            nt = time.perf_counter()
            print('iteration runtime', nt - ct, 'runtime per word', (nt - ct) / gmetric)

            oldslice = newslice
            newslice += gmetric
            if gmetric == 100: gmetric = 0
            gmetric += siftcount * 8

        uk = [] #Get rid of any that were posts added before they could be populated.
        for k, v in Head.children.items():
            if not v.vector:
                uk.append(k)
        for k in uk:
            del Head.children[k]
        Head = self.package(Head, context)
        heads.append(Head)

        #Need to make sure we packaged any remnants of our run before doing the fold.
        if len(heads) > 0:
            # Then we want to take all the heads, and convert them into a single domain, which we can then iteratively transform by level.
            domains = Domain(1)
            ks = []
            for i, head in enumerate(heads):
                for k,v in head.children.items():
                    domains.adopt(v.name,v)
                    if v.age > 60: ks.append((v.name[3:], v.vector)) #Test with the dynamic domains tomorrow ##############
            counter = 1
            for domain in domains.children.values():
                domain.dtargets = domain.dtargets + ks
                domain.keystone()
            while len(domains.children.keys()) > 4:
                print("Doing another fold", counter)
                counter += 1
                domains, domainlevel = self.fold(domains,context, -.4)
                domains, totalintegrity, domaincount = self.integrity(domains, context)
                domains = self.package(domains, context)

        return domains

    def integrity(self, domain, context):
        supertotal = []
        dcount = 0
        for k, child in domain.children.items():
            if not child.vector: continue
            dcount += 1
            if child.level == 0:
                ptable = self.dprobs(child.vector, context) #Right here, I think is the one that's making the fold take so long, as it ascends those vectors become huge
                totals = [] #cont. we should be able to circumvent by simply checking the child's integrity, which I think already exists.
                for k, v in ptable.items():
                    total = sum([score for score in v.values()])
                    totals.append(total)
                child.integrity = sum(totals) / len(totals)
                supertotal.append(sum(totals) / len(totals))
            else:
                supertotal.append(child.integrity)
        domain.integrity = sum(supertotal) / len(supertotal)
        return domain, domain.integrity, dcount

    def findposts(self, pwords, postcount: int, context, dtol, siftcount) -> list:
        # build domain and randomly populate
        Head = Domain(1)
        counter = 0
        rwords = pwords.copy()
        random.shuffle(rwords)

        partitions = self.findpartitions(pwords, postcount)
        scope = []
        for i, value in enumerate(rwords):
            scope.append(value[0])
            if i == partitions[counter]:
                Head.grow(scope)
                scope = []
                counter += 1
        ptable = self.dprobs(pwords, context)
        # Stage 2
        #posts = self.dtop_trans(Head)
        Head = self.sifter(Head, ptable, context, 50, siftcount, postcount, dtol)
        # domain = self.ptod_trans(posts, domain)
        return Head

    def findpartitions(self, pwords, quantity):
        partitions = []
        dnum = range(1, quantity + 1)
        base = len(pwords) // quantity
        for num in dnum:
            if num == dnum[-1]:
                partitions.append(len(pwords) - 1)
            else:
                partitions.append((base * num) - 1)
        return partitions

    def dprobs(self, pwords, context):
        ptable = {}
        for upper in pwords:
            if type(upper) != str:
                upper = upper[0]
            ptable[upper] = []
            ltable = {}
            for lower in pwords:
                if type(lower) != str:
                    lower = lower[0]
                if upper == lower:
                    continue
                gold, silver = self.vect_to_num(context[upper], context[lower])
                similarity = np.log(self.cosine(gold, silver))
                ltable[lower] = similarity
            ptable[upper] = ltable
        return ptable

    def domain_d_measure(self, domain):
        dcontext = {}
        for k, v in domain.children.items():
            dcontext[k] = v.dcontext

        #Calculate distance between domains
        print("Calculating a new set of domain distances")
        distance = {}
        for k, v in dcontext.items():
            sd = {}
            for k2, v2 in dcontext.items():
                if v == v2:
                    continue
                gold, silver = self.vect_to_num(v, v2)
                similarity = np.log(self.cosine(gold, silver))
                sd[k2] = similarity
            distance[k] = sd

        return distance, dcontext

    def growpost(self, domain, pwords, quantity, context, dtol):
        additions = pwords.copy()
        for k, child in domain.children.items():
            for word in child.vector:
                pwords.append((word, 'Domain:{}'.format(k)))
        ptable = self.dprobs(pwords, context)
        keys = list(domain.children.keys())
        partitions = self.findpartitions(pwords, len(keys))
        for child in domain.children.values():
            while len(additions) > 0 and len(child.vector) < partitions[0]:
                child.vector.append(additions.pop()[0])

        while len(additions) > 0:
            domain.children[keys[0]].vector.append(additions.pop()[0])
            random.shuffle(keys)
        domain = self.sifter(domain, ptable, context, 50, 4, quantity, dtol)
        return domain

    def package(self, domain, context):
        dcontext = {}
        for k, v in domain.children.items():
            subdomain = {}
            for word in v.vector:
                subdomain[word] = context[word]
            dcontext[k] = dict(self.reshape(subdomain))
            if v.name == None: v.name = self.findname(v, dcontext[k], context)
            v.dcontext = dcontext[k]
            v.identity = False
        return domain

    def fold(self, domain, context, tol):
        level = domain.level
        dtable, dcontext = self.domain_d_measure(domain)
        assigned = []
        groupings = []
        for k, v in dtable.items():
            if k in assigned: continue
            closest = list(v.items())
            closest.sort(key=lambda x: x[1], reverse=True)
            grouping = [k]
            assigned.append(k)
            for cl in closest: #Rebuild here, so that I can end up with the amount of groupings I want.
                if cl[0] in assigned: continue
                grouping.append(cl[0])
                assigned.append(cl[0])
                if cl[1] < tol: break
                if len(grouping) == 4: break
            groupings.append(grouping)
            tol -= .05


        newhead = Domain(level + 1)
        for group in groupings:
            nd = Domain(level)
            nd.identity = True
            supervector = {}
            for dg in group:
                for child in domain.children.values():
                    if child.name == dg:
                        if child.identity == True: continue #This is to avoid reassigning a domain that's already assigned.
                        child.identity = True
                        nd.children[dg] = child
                        nd.vector = nd.vector + child.vector
                        supervector = {**supervector, **dcontext[dg]}
            nd.name = self.findname(nd, supervector, context)
            nd.keystone()
            newhead.children[nd.name] = nd
            newhead.vector = newhead.vector + nd.vector
        newhead.keystone()
        return newhead, newhead.level

    def findname(self, domain, supervector, context):
        if not domain.vector: return 'UNKNOWN'
        name = []
        for word in domain.vector:
            gold, silver = self.vect_to_num(context[word], supervector)
            sim = self.cosine(gold, silver)
            name.append((sim, word))
        name.sort()
        final = ""
        for i in range(domain.level+1):
            final = final + "." + name[-(i+1)][1]
        return str(domain.level) + "." + final

    def sifter(self, domain, ptable, context, epochs: int, siftcount: int, quantity, dtol: int):
        # Change sifter to accept a domain, we can call the sifter multiple times from outside.
        epochs -= 1
        if epochs == 0 or siftcount == 0:
            return domain
        else:
            dprobs = {}
            for k, v1 in domain.children.items():
                v1.age += 1
                wprob = []
                for w1 in v1.vector:
                    sim = 0
                    for w2 in v1.vector:
                        if w1 == w2: continue
                        sim += ptable[w1][w2]
                    wprob.append((w1, sim))
                dprobs[k] = wprob

            # Get the probabilities for every word as referenced against every other word in its domain. Now time to evict the lowest probability.
            reorder = list(domain.children.keys())
            random.shuffle(reorder)
            for k, v in dprobs.items():
                v.sort(key=lambda x: x[1], reverse=True)
                total = sum([val[1] for val in v])
                self.dtotals.append((k, total, 'epoch', epochs))
                for worst in v[-siftcount:]:
                    domain.children[k].vector.remove(worst[0]) #Wow, now that's getting long -> more complex to make it simple :D
                    popped = reorder.pop(0)
                    reorder.append(popped)
                    if worst[0] in self.rejects:
                        self.rejects[worst[0]] += 1
                    else:
                        self.rejects[worst[0]] = 1

                    if self.rejects[worst[0]] < 20:
                        domain.children[reorder[0]].vector.append(worst[0])
                    else:  # This is the way we drop words that aren't part of any of the x domain
                        self.gcounter += 1
                        if self.gcounter == dtol:
                            self.gcounter = 0
                            siftcount -= 1
                        if siftcount == 0:
                            #domain.children[reorder[0]].vector.append(worst[0])
                            domain = self.balancer(domain, ptable, len(domain.children))
                            return domain
            totalprob = sum([val[1] for val in self.dtotals[-quantity:]])
            wordcount = sum([len(v.vector) for v in domain.children.values()])
            self.totprob.append(("totalprob:", totalprob, "avgwrdprob", totalprob / wordcount, "totalwordcount", wordcount))
            domain = self.sifter(domain, ptable, context, epochs, siftcount, quantity, dtol)
        return domain

    def isbalanced(self, domain, dsize):
        for v in domain.children.values():
            if len(v.vector) > dsize + 4: return False
            if len(v.vector) < dsize - 4: return False
        return True

    def balancer(self, domain, ptable, epochs):
        dsize = sum(len(v.vector) for v in domain.children.values())
        dsize = dsize // len(domain.children)
        epochs -= 1
        if self.isbalanced(domain, dsize) or epochs == 0: return domain #
        # Find the worst words in each domain.
        dprobs = {}
        for k, v1 in domain.children.items():
            wprob = []
            for w1 in v1.vector:
                sim = 0
                for w2 in v1.vector:
                    if w1 == w2: continue
                    sim += ptable[w1][w2]
                wprob.append((w1, sim))
            dprobs[k] = wprob
        worst = {}
        for k, v in dprobs.items():  # Now let's connect the worst
            v.sort(key=lambda x: x[1], reverse=True)
            worst[k] = [wv[0] for wv in v[-epochs:]]
            total = sum([val[1] for val in v])
            self.dtotals.append((k, total, 'epoch', epochs))
        swapvalue = {}
        subdict = {k: 0 for k in domain.children.keys()}
        for wlist in worst.values():
            for word in wlist:
                swapvalue[word] = subdict.copy()
                for k, v1 in domain.children.items():
                    for wv1 in v1.vector:
                        if word == wv1: continue
                        swapvalue[word][k] += ptable[word][wv1]

        for k, v in domain.children.items():
            if len(v.vector) <= dsize: continue
            target = worst[k].pop(-1)
            domain.children[k].vector.remove(target)
            newhome = list(swapvalue[target].items())
            newhome.sort(key=lambda x: x[1], reverse=True)
            newhome = str(newhome[0][0])
            domain.children[newhome].vector.append(target)

        domain = self.balancer(domain, ptable, epochs)
        return domain

    def groupings(self, target, context) -> Grouping:
        groupings = {}
        vector = context[target]
        for key in vector.keys():
            numprobs = []
            vecroot = key
            for key2 in vector.keys():
                if key != key2:
                    gold, silver = self.vect_to_num(context[key], context[key2])
                    probability = self.cosine(gold, silver)
                    probability = float(probability)
                    numprobs.append((key2, probability))
            numprobs.sort(key=lambda x: x[1], reverse=True)
            groupings[vecroot] = numprobs
        return groupings

    def reshape(self, target: Target) -> Supervector:
        sv = {}
        for val in target.values():
            for k, v in val.items():
                if k == 'stopword':
                    continue
                if k in sv:
                    sv[k] += v
                else:
                    sv[k] = v
        sv = list(sv.items())
        sv.sort(key=lambda x: x[1], reverse=True)
        return sv

    def recycle(self, rejects: dict, recyclecount):
        keys = rejects.keys()
        recycled = []
        for key in keys:
            if rejects[key] == 20:
                if key not in self.lastchance:
                    self.lastchance[key] = 1
                    rejects[key] = 0
                    recycled.append(key)
                else:
                    self.lastchance[key] = 'DEAD'
            if len(recycled) == recyclecount:
                return recycled
        return recycled

    def cosine(self, array1, array2):
        numerator = sum(array1 * array2)
        # print(numerator)
        denominator = sum(array1 ** 2) ** .5 * sum(array2 ** 2) ** .5
        # print(denominator)
        if numerator and denominator != 0:
            return numerator / denominator
        else:
            return .0000001

    def vect_to_num(self, vect1, vect2):
        # Should be receiving a dict, which is transformed list of tups [(word, count)]
        v1dict = {}
        v2dict = {}
        vect1 = vect1.items()
        vect2 = vect2.items()

        for pair1 in vect1:
            v1dict[pair1[0]] = pair1[1]
            v2dict[pair1[0]] = 0

        for pair2 in vect2:
            if pair2[0] in v2dict:
                v2dict[pair2[0]] += pair2[1]
            else:
                v2dict[pair2[0]] = pair2[1]
            if pair2[0] not in v1dict:
                v1dict[pair2[0]] = 0

        v1list = list(v1dict.items())
        v2list = list(v2dict.items())
        assert len(v1list) == len(v2list)

        v1list.sort(key=lambda x: x[0])  # MUST sort alphabetical, sorting by value would ruin it.
        v2list.sort(key=lambda x: x[0])

        v1array = np.zeros(len(v1list))
        v2array = np.zeros(len(v2list))

        for i in range(len(v1list)):
            v1array[i] = v1list[i][1]
            v2array[i] = v2list[i][1]

        return v1array, v2array
