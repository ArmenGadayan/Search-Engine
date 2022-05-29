import re
from bs4 import BeautifulSoup
import nltk
import json
import os
import pickle
import sys
import hashlib
import math

# Partial Inverted Index Dictionary 
partialIndex = {}
# Array to hold all the webpage urls
docIDMap = []

lexicon = {}

fingerprints = {}

def createIndex(path):
    global partialIndex, docIDMap, fingerprints
    nltk.download("punkt")

    rootdir = path

    # Doc ID initialized to 0
    docID = 0
    counter = 0
    # Traverse through all the files
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            path = os.path.join(subdir, file)
            counter += 1
            print("counter", counter)
            # Load json data of each page into variable data
            with open(path) as f:
                data = json.load(f)

            # Content of each page
            content = data['content']
            #URL of each page
            url = data['url']

            # Parse html
            soup = BeautifulSoup(content, 'html.parser')

            # Get tokens with our custom tokenize funtion
            tokens = tokenize(soup)

            # Get simhash fingerprint of document
            fingerprint = simhash(tokens)
            skip = False
            for fp in fingerprints.values():
                # If fingerprint is similar to an existing fingerprint, ignore document
                if is_similar(fingerprint, fp):
                    skip = True
            if skip: 
                continue

            # If fingerprint is unique, add to list of fingerprints
            fingerprints[str(fingerprint)] = fingerprint
            
            docLength = docLen(tokens)

            # Traverse through each token
            for token in tokens:
                # Add new key token in the inverted index if the token is new
                if token not in partialIndex:
                    partialIndex[token] = [(docID, tokens[token], docLength)]
                # Add onto the already existing token with new DOCID and term frequency tuple
                else:
                    partialIndex[token].append((docID, tokens[token], docLength))
            # Add to the list of the webpage urls. Note: docIDMap[docID] is the url of the webpage with that id 
            docIDMap.append(url)

            docID += 1
            if docID % 8000 == 0:
                savePartialIndex()
                print("saved partial index")
                
    savePartialIndex()
    save_globals(docID)

def savePartialIndex():
    global partialIndex, lexicon
    f = open("savedData/InvertedIndex.txt", "a+")
    for token in partialIndex:
        if token not in lexicon:
            lexicon[token] = [f.tell()]
        else:
            lexicon[token].append(f.tell())
        f.write(json.dumps(partialIndex[token]))
        f.write("\n")

    f.close()
    partialIndex.clear()

def tokenize(soup):
    tokens = []
    #create dictionary to hold tokens and their frequencies
    frequencies = {}     
  
    stemmer = nltk.stem.PorterStemmer()
    for p in soup.find_all(True, text=True):
        tag = p.name
        words = nltk.tokenize.wordpunct_tokenize(p.get_text())
        for word in words:
            token = stemmer.stem(word).lower()
            if token not in frequencies: frequencies[token] = 1
            else: frequencies[token] += 1
            # Weigh important tokens with higher frequency
            if tag not in ["a", "p"]:
                if tag == "title": frequencies[token] += 100
                elif tag == "h1": frequencies[token] += 50
                elif tag == "h2": frequencies[token] += 45
                elif tag == "h3": frequencies[token] += 40
                elif tag == "b": frequencies[token] += 10
                elif tag == "strong": frequencies[token] += 10

    return frequencies

  
def save_globals(lastDocID):
    global docIDMap, lexicon

    f = open("savedData/DocIDMap.txt", "wb+")
    pickle.dump(docIDMap, f)
    f.close()

    f = open("savedData/Lexicon.txt", "wb+")
    pickle.dump(lexicon, f)
    f.close()

def docLen(tokens):
    length = 0
    for token in tokens:
        weight = 1 + math.log(tokens[token], 10)
        length += weight*weight
    return math.sqrt(length)

def simhash(tokens):
    #create array of 32 ints to store the result of weighing all tokens
    simhash = [0] * 32
    for token in tokens:
        #encode tokens in utf-8
        tokenHash = hashlib.md5(token.encode()).digest()[:4]
        bitmask = 0x1
        #for each bit in hash value add the frequency if the bit is 1 or subtract if 0
        for j in range(32):
            if int.from_bytes(tokenHash, "big") & bitmask > 0:
                simhash[j] += tokens[token]
            else:
                simhash[j] -= tokens[token]
            bitmask = bitmask << 1

    #convert simhash array into 32-bit fingerprint
    setbit = 0x1
    fingerprint = 0x0
    #for each int in simhash, if int is > 0 set bit to 1
    for i in range(32):
        if simhash[i] > 0:
            fingerprint = fingerprint | setbit
        setbit = setbit << 1
    
    return fingerprint
        
def is_similar(fingerprint1, fingerprint2):
    #XOR bits of the two fingerprints and invert, then count number of 1's to get number of similar bits
    sum_of_same = bin((fingerprint1^fingerprint2)^0xffffffff).count("1")
    #divide number of similar bits by total number of bits to get similarity
    similarity = sum_of_same/32
    #return true if similarity is greater than threshold
    return similarity >= 0.95
