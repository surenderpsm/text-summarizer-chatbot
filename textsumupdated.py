import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
# function to read the article and return an 2D list with all the sentences
def read_art(fname):
    fname = open(fname,"r")
    fdata = fname.readlines()
    article = fdata[0].split(". ")
    sentences=[]
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]"," ").split(" "))
    sentences.pop()
    return sentences
# function to compare 2 sentences and rank them
def s_sim(s1,s2,stopwords=None):
    if stopwords is None:
        stopwords=[]
    s1 = [w.lower() for w in s1]
    s2 = [w.lower() for w in s2]
    allw = list(set(s1+s2))
    
    v1 = [0] * len(allw)
    v2 = [0] * len(allw)
    for w in s1:
        if w in stopwords:
            continue
        v1[allw.index(w)] += 1
    for w in s2:
        if w in stopwords:
            continue
        v2[allw.index(w)] += 1
    return 1-cosine_distance(v1,v2)
# function to generate the sentence matrix
def sim_matrix(sentences,stop_words):
    sim_mat = np.zeros((len(sentences),len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1==idx2:
                continue
            sim_mat[idx1][idx2]=s_sim(sentences[idx1],sentences[idx2],stop_words)
    return sim_mat
# generate summary
def gensum(fname,n=5):
    stop_words=stopwords.words('english')
    summarize_text=[]
    sentences = read_art(fname)
    sentence_sim_matrix = sim_matrix(sentences,stop_words)
    sentence_sim_graph = nx.from_numpy_array(sentence_sim_matrix)
    scores = nx.pagerank(sentence_sim_graph)
    ranked_sentence = sorted(((scores[i],s)for i,s in enumerate(sentences)),reverse=True)
    for i in range(n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    print("Summary \n",". ".join(summarize_text))
val = input("Enter size of summary: ")

gensum("nogap.txt", int(val))
