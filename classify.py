#!/usr/bin/env python

# Akshay Srivatsan

# Usage:
#
# ./classify.py plant.tokenized plant.titles
#

import sys
import random
from collections import defaultdict

NUM_DOCS = 4000
NUM_TRAIN = 3600

def main():
    # open the files
    inFile = open(sys.argv[1])
    titleFile = open(sys.argv[2])
    suffix = ""
    if "stemmed" in sys.argv[1]:
        suffix = ".stemmed"
    commonFile = open("common_words" + suffix)

    # track the true sense
    sensenum = [0] * NUM_DOCS
    # store the titles
    titles = [""] * NUM_DOCS
    # store the "document" vectors
    vecs = []
    # store stopwords
    common = []

    # read in common words
    for line in commonFile:
        line = line.strip()
        common.append(line)

    # read in the titles
    for line in titleFile:
        elements = line.split()
        sentID = int(elements[0])

        sensenum[sentID - 1] = int(elements[1])
        titles[sentID - 1] = line

    # read in the data
    currDoc = 0
    sentence = []
    for line in inFile:
        line = line.strip()
        words = line.split()

        # if new doc
        if len(words) > 0 and words[0] == ".I":
            # process previous sentence
            processSentence(sentence, vecs, common, currDoc)

            # rinse and repeat
            sentence = []
            currDoc = int(words[1])
            vecs.append(defaultdict(lambda: 0))
            continue

        sentence.append(line)
    # run again for the last one
    processSentence(sentence, vecs, common, currDoc)

    # find centroids
    # replace with findCentroids(vecs, sensenum) for parts 1-3
    (v_profile1, v_profile2) = kmeans(vecs, sensenum)

    # evaluate test data
    correct = 0
    incorrect = 0
    for doc in range(NUM_TRAIN, NUM_DOCS):
        sim1 = similarity(vecs[doc], v_profile1)
        sim2 = similarity(vecs[doc], v_profile2)

        sense = 0
        if sim1 > sim2:
            sense = 1
        else:
            sense = 2

        result = ""
        if sense == sensenum[doc]:
            result = "+"
            correct += 1
        else:
            result = "*"
            incorrect += 1

        print "%s\tSim1: %.4e Sim2: %.4e Diff: %.4e\t%s" % \
              (result, sim1, sim2, sim1 - sim2, titles[doc].strip())
    print "Percentage accuracy: %f" % (float(correct) / (correct + incorrect))

def processSentence(sentence, vecs, common, currDoc):
    # locate keyword
    keyword = ""
    for word in sentence:
        if ".x-" in word or ".X-" in word:
            keyword = word
    # build the vector
    for word in sentence:
        # add to term vector if not a stopword or the keyword
        if word not in common and word != keyword:
            '''
            # check if word is immediately to right or left of keyword
            dist = sentence.index(word) - sentence.index(keyword)
            if dist  == 1:
                sentence[sentence.index(word)] = "R-" + sentence[sentence.index(word)]
                word = "R-" + word
            elif dist == -1:
                sentence[sentence.index(word)] = "L-" + sentence[sentence.index(word)]
                word = "L-" + word
            '''

            # switch out uniformDecay, expDecay, steppedDecay, or linearDecay to try out different weightings
            vecs[currDoc - 1][word] += linearDecay(sentence, word, keyword)

def linearDecay(sentence, word, keyword):
    index = sentence.index(word)
    keyIndex = sentence.index(keyword)
    if index < keyIndex:
        return float(index + 1) / keyIndex
    else:
        return float(len(sentence) - index) / (len(sentence) - 1 - keyIndex)

def steppedDecay(sentence, word, keyword):
    index = sentence.index(word)
    keyIndex = sentence.index(keyword)
    dist = abs(index - keyIndex)
    if dist == 1:
        return 6
    elif dist == 2 or dist == 3:
        return 3
    else:
        return 1

def expDecay(sentence, word, keyword):
    index = sentence.index(word)
    keyIndex = sentence.index(keyword)
    dist = abs(index - keyIndex)
    return float(1) / dist

def uniformDecay(sentence, word, keyword):
    return 1

# use k means clustering to find unsupervised centroids
def kmeans(vecs, sensenum):
    # initialize centroids randomly
    v_profile1 = vecs[random.randrange(0, NUM_TRAIN)]
    v_profile2 = vecs[random.randrange(0, NUM_TRAIN)]

    # keep track of assignment of each document
    assignment = [0] * NUM_TRAIN

    pigs = "pigs"
    sys.stdout.write("Calculating K-Means Clusters")
    while pigs != "fly":
        sys.stdout.write(".")
        sys.stdout.flush()
        # track when we make an update so we can terminate at stability
        pigs = "fly"

        # assign each document to a centroid
        maxSim1 = float("-inf")
        mostSim = -1
        for doc in range(0, NUM_TRAIN):
            prevAssignment = assignment[doc]
            sim1 = similarity(vecs[doc], v_profile1)
            if sim1 > similarity(vecs[doc], v_profile2):
                assignment[doc] = 1
                # track the closest document to centroid 1
                if maxSim1 < sim1:
                    maxSim1 = sim1
                    mostSim = doc
            else:
                assignment[doc] = 2

            # if we updated at least one doc, continue
            if prevAssignment != assignment[doc]:
                pigs = "pigs"

        # calculate new centroids
        (v_profile1, v_profile2) = findCentroids(vecs, assignment)

    print

    # check if semantics are switched
    # use the closest document to centroid 1 to label
    if sensenum[mostSim] != 1:
        tmp = v_profile1
        v_profile1 = v_profile2
        v_profile2 = tmp

    return (v_profile1, v_profile2)

def findCentroids(vecs, sensenum):
    v_profile1 = defaultdict(lambda: 0)
    v_profile2 = defaultdict(lambda: 0)
    num1 = 0
    num2 = 0

    # add up term weights from training data
    for doc in range(0, NUM_TRAIN):
        if sensenum[doc] == 1:
            num1 += 1
            for term in vecs[doc].keys():
                v_profile1[term] += vecs[doc][term]
        else:
            num2 += 1
            for term in vecs[doc].keys():
                v_profile2[term] += vecs[doc][term]

    # divide by number of documents
    for term in v_profile1:
        v_profile1[term] = float(v_profile1[term]) / num1
    for term in v_profile2:
        v_profile2[term] = float(v_profile2[term]) / num2

    return (v_profile1, v_profile2)

# cosine similarity for two vectors
def similarity(vec1, vec2):
    num = 0
    sum_sq1 = 0
    sum_sq2 = 0

    val1 = vec1.values()
    val2 = vec2.values()

    # determine shortest vector
    if len(val1) > len(val2):
        tmp = vec1
        vec1 = vec2
        vec2 = tmp

    # calculate cross product
    for key in vec1.keys():
        num += vec1[key] * vec2[key]

    # calculate sum of squares
    for term in val1:
        sum_sq1 += term * term
    for term in val2:
        sum_sq2 += term * term

    return float(num) / ((sum_sq1 * sum_sq2) ** 0.5)

if __name__ == "__main__":
    main()
