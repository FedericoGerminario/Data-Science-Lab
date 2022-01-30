import json
from mlxtend.frequent_patterns import apriori, fpgrowth
import pandas as pd
from timeit import timeit

#functions
def generateFirstItemset(dataset):

    L1 = {}
    for it in dataset:
        for el in it:
            elm = (el,)
            if L1.get(elm, 0) != 0:
                L1[elm] += 1
            else:
                L1[elm] = 1
    return L1

def storeFrequentItemset(L, data, minsup):
    Lk = {k: v for k,  v in data.items() if v >= minsup}
    Lnk = {k: v for k, v in data.items() if v < minsup}
    if len(Lk) > 0:
        L.append(Lk)
    return list(Lk.keys()), list(Lnk.keys())

def generateCandidates(Lk):
    Ck = []
    for i in range(len(Lk)):
        for j in range(i+1, len(Lk)):
            if Lk[i][: -1] == Lk[j][: -1]:
                Ck.append(Lk[i] + (Lk[j][-1],))
    return Ck

def pruneCandidates(Ck, Lnk):
    C = []
    flag = 0
    for i in range(len(Ck)):
        for j in range(len(Lnk)):
            lenght = 0
            for n in (Ck[i]):
                if n in Lnk[j]:
                    lenght += 1

            if lenght == len(Lnk[j]):
                lenght = 0
                flag = 1
                break
            else:
                lenght = 0

        if flag != 1:
            C.append(Ck[i])

        else:
            flag = 0

    return C

def scanCandidates(dataset, Lk):
    L = {}
    for el in Lk:
        for al in dataset:
            lenght = 0
            for en in el:
                if en in al:
                    lenght += 1

            if lenght == len(el):
                if L.get(el, 0) != 0:
                    L[el] += 1
                else:
                    L[el] = 1
                lenght = 0
    return L

def newApriori(dataSet, minsup):
    L = []
    L1 = generateFirstItemset(dataSet)
    Lk, Lnk = storeFrequentItemset(L, L1, minsup)
    while len(Lk) != 0:
        Ck = generateCandidates(Lk)
        Ck = pruneCandidates(Ck, Lnk)
        data = scanCandidates(dataSet, Ck)
        Lk, Lnk = storeFrequentItemset(L, data, minsup)

    return L




minsup = 1
dataSet = [['a', 'b'],
           ['b', 'c', 'd'],
           ['a', 'c', 'd', 'e'],
           ['a', 'd', 'e'],
           ['a', 'b', 'c'],
           ['a', 'b', 'c', 'd'],
           ['b', 'c'],
           ['a', 'b', 'c'],
           ['a', 'b', 'd'],
           ['b', 'c', 'e'],
           ['f']
          ]

#L = apriori(dataSet, minsup)
#print(L)

with open('modifiedCoco.json') as f:
    images = json.load(f)
datasetCoco = [list(set(image['annotations'])) for image in images ]


a = newApriori(datasetCoco, 100)

#Modify dataset to fit in apriori and fpgrowth
allItemSet = set()
for items in datasetCoco:
    allItemSet.update(items)
allItems = sorted(list(allItemSet))
presenceMatrix =[[int(item in image) for item in allItems]for image in datasetCoco]

df = pd.DataFrame(presenceMatrix, columns=allItemSet)
fiAp = apriori(df, 0.02)
fiFp = fpgrowth(df, 0.02)
tuplesAp= {tuple(row) for row in fiAp.values}
tuplesFp = {tuple(row) for row in fiFp.values}
fi_myap = set()
for i in range(len(a)):
    for Lk in a[i]:
        fi_myap.update({(v/5000, frozenset({allItems.index(k_) for k_ in k}))for k, v in a[i].items()})


print(a[1])
print(tuplesAp)
print(fi_myap)
ugual = tuplesAp == fi_myap
print("fpgrowth", timeit(lambda: fpgrowth(df, 0.02), number=1))
print("apriori", timeit(lambda: apriori(df, 0.02), number=1))
print("newApriori", timeit(lambda: newApriori(datasetCoco, 0.02 * 5000), number=1))
