# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:38:30 2019

@author: Mina Ekramnia

"""
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import docx
import argparse
import os
import wget
import zipfile
from argparse import ArgumentParser


def read_article(filepath):
    # IF the file is .csv format
    if '.csv' in filepath:
        df = pd.read_csv(r"filepath")
    elif '.docx' in filepath:
        df = docx.Document(r'filepath')   # array of string
    elif '.txt' in filepath:
        with open(filepath) as f:
            text = f.read() 
        return text   # early return of an string

    text = []
    for i in df:
        text.append(i.text)
    return text   # array


def get_glove_vectors():
    # check for file if it exists
    # download the zipfile and extract in to txts
    path_to_file = 'glove_vectors.zip'
    if not os.path.exists(path_to_file):
        wget.download('http://nlp.stanford.edu/data/glove.6B.zip', path_to_file)

    with zipfile.ZipFile(path_to_file) as zip:
        zip.extractall('glove_vectors')

    word_embeddings = {}
    with open("glove_vectors/glove.6B.50d.txt", encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = coefs

    return word_embeddings


def clean_sentences(sentences):
    stop_words = set(stopwords.words('english'))

    new_sentences = []
    for sentence in sentences:
        lowercase_sentence = sentence.lower().replace('.', '').replace(',', '')
        tokenized_sentence = lowercase_sentence.split(' ')
        new_sentences.append([word for word in tokenized_sentence if word not in stop_words])

    return new_sentences


def get_sentence_vectors(sentences):

    cleaned_sentences = clean_sentences(sentences)

    embedding_dict = get_glove_vectors()

    vectorized_setences = []
    for sentence in cleaned_sentences:
        word_vectors = [] 
        for word in sentence:
            vec = embedding_dict.get(word, np.zeros((50,)))
            word_vectors.append(vec)  #list of 50 dim vectors
            #  to represent each sentence by taking avg of all word vectors    
        sentence_vector = sum(word_vectors, 0)/len(word_vectors)

        vectorized_setences.append(sentence_vector)

    return vectorized_setences


def get_top_ranked_sentences(sentence_vectors, n):
    # Sort the rank and pick top sentences based on Similarity matrix
    # We will use Cosine Similarity to compute the similarity between a pair of sentences
    sim_mat = np.zeros([len(sentence_vectors), len(sentence_vectors)])
    for i in range(len(sentence_vectors)):
        for j in range(len(sentence_vectors)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,50), sentence_vectors[j].reshape(1, 50))[0, 0]      
    #  we represent a graph by adjancacy matrix.
    #  score based on those sentences that are more relevant
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    print(nx_graph)

    #  A tuple of two elements: index and scores.
    ranked_indexes = [key for (key, value) in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    #  Extract top 3 sentences as the summary
    return ranked_indexes[:n]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()   # initializing parser obj
    #  parser = argumentparser()
    parser.add_argument('filename')   # default=3 n,type=int calling a method
    parser.add_argument('--n', type=int, default=3, help='add number of sentences')
    #  parser = argparse.ArgumentParser(description='Process some integers.')
    args = parser.parse_args()  # give me all the arguments
    #  filename = input('Please enter the filename: ')
    text = read_article(args.filename)
    sentences = sent_tokenize(text)
    sentence_vectors = get_sentence_vectors(sentences)
    ranked_indexes = get_top_ranked_sentences(sentence_vectors, args.n)   # return indexes
    top_sentences = [sentences[i] for i in ranked_indexes]   # a list of sentences in to one string.
    summary = ' '.join(top_sentences)
    print(summary)