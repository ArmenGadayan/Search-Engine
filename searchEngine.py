from InvertedIndex import createIndex
from os.path import exists
import pickle
import json
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import PorterStemmer
import nltk
import math
import time
from bisect import bisect_left
from heapq import nlargest

class SearchEngine:
    def __init__(self):
        if not exists("savedData/InvertedIndex.txt"):
            createIndex("DEV")
        self.invertedIndex = open("savedData/InvertedIndex.txt", "r")
        with open("savedData/DocIDMap.txt", "rb") as f:
            self.docIDMap = pickle.load(f)
        with open("savedData/Lexicon.txt", "rb") as f:
            self.lexicon = pickle.load(f)
        self.numDocuments = len(self.docIDMap)
        

    def search(self, query):
        stemmer = PorterStemmer()
        #stem and lowercase the query terms
        words = [stemmer.stem(word.lower()) for word in wordpunct_tokenize(query)]
        
        dfScores = {}
        postings = {}
        
        #get the posings for each query term from the inverted index and store them in a dictionary
        for word in words:
            posting = self.getPostings(word)
            postings[word] = posting 
            dfScores[word] = len(posting)
        
        #return the top ranked documents
        return self.cosineScores(words, dfScores, postings)
                        
    def binarySearch(self, arr, low, high, x):
        if high >= low:
            mid = (high + low) // 2
            if arr[mid][0] == x:
                return mid
            elif arr[mid][0] > x:
                return self.binarySearch(arr, low, mid - 1, x)
            else:
                return self.binarySearch(arr, mid + 1, high, x)
        else:
            return -1

    #return raw tf score and divided by document lenght
    def tf(self, term, doc, postings):
        document = self.findDoc(term, doc, postings)
        if document:
            freq = document[1]
            docLen = document[2]
            return freq/docLen
        else:
            return 0

    #return inverse document frequency
    def idf(self, df):
        if df != 0:
            return  math.log(self.numDocuments/df, 10)
        else:
            return 0
    
    #return a positng of a given document
    def findDoc(self, term, docid, postings):
        i = self.binarySearch(postings, 0, len(postings)-1, docid)
        if i != -1:
            return postings[i]
        else:
            return None

    #return the postings of a term by seeking in the inverted index file
    def getPostings(self, token):
        postings = []
        try:
            for position in self.lexicon[token]:
                self.invertedIndex.seek(position)
                postings.extend(json.loads(self.invertedIndex.readline()))
            return postings
        except KeyError:
            return postings

    #return the top 50 documents based on their cosine scores
    def cosineScores(self, q, dfScores, postings):
        scores = {}
        #get all documents that either match all query terms or match all but one query term
        potentialDocs = self.getPotentialDocs(q, 50, postings)
        #store query as dictionary with query term frequency as values
        q = self.getFreq(q)
        weightQ = {}
        qNormLength = 0
        
        #calculate the tf-idf for each query term
        for term in q:
            df = dfScores[term]
            tf = 1 + math.log(q[term], 10)
            weight = tf*self.idf(df)
            weightQ[term] = weight
            qNormLength += weight*weight
        qNormLength = math.sqrt(qNormLength)
        
        #normalize the query tf-idf
        for term in q:
            if qNormLength != 0:
                weightQ[term] = weightQ[term]/qNormLength

        #calculate the normalized tf for each term for each document
        for term in q:
            for docid in potentialDocs:
                score = self.tf(term, docid, postings[term])
                if docid not in scores:
                    scores[docid] = score*weightQ[term]
                else:
                    scores[docid] += score*weightQ[term]

        #convert scores dictionary to a list
        scoreList = scores.items()
        #retrieve top 50 scores using maxheap
        topDocs = nlargest(50, scoreList, key=lambda r:r[1])
        results = []
        #convert list of docids to list of urls
        for doc in topDocs:
            results.append(self.docIDMap[doc[0]])
        return results

    #return a dictionary of frequencies given list of words
    def getFreq(self, words):
        freq = {} 
        for word in words:
            if word not in freq:
                freq[word] = 1 
            else:
                freq[word] += 1
        return freq
    
    #return all documents that either match all query terms or match all but one query term
    def getPotentialDocs(self, query, k, postings):
        potentialIDS = {}
        #get all documents that match all query terms
        for word in query:
            try:
                for docid in postings[word]:
                    if docid[0] not in potentialIDS:
                        potentialIDS[docid[0]] = 1
                    else:
                        potentialIDS[docid[0]] += 1
            except KeyError:
                print("Doesn't exist")
        
        size = len(query)
        extraDocs = {}
        #get all documents that match all but one query term
        for docid in potentialIDS.copy():
            if size != 1 and potentialIDS[docid] == size-1:
                extraDocs[docid] = potentialIDS[docid]

            if potentialIDS[docid] != size:
                potentialIDS.pop(docid, None)
                
        #if k documents aren't exact matches, add extra docs that are missing one term
        if len(potentialIDS) < k and size != 1:
            potentialIDS.update(extraDocs)
        return potentialIDS
