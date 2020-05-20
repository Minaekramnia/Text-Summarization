import os
import wget
from argparse import ArgumentParser
import zipfile
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity


GLOVE_VECTORS_URL = 'http://nlp.stanford.edu/data/glove.6B.zip'


def get_setences_from_file(filename):
    with open(filename) as f:
        text = f.read()

    return sent_tokenize(text)


def get_word_vectors():
    if not os.path.exists('glove_vectors'):
        print('Downloading glove vectors')
        wget.download(GLOVE_VECTORS_URL, 'glove.6b.zip')
        with zipfile.ZipFile('glove.6b.zip', 'r') as zip_ref:
            zip_ref.extractall('glove_vectors')
    else:
        print('Loading glove vectors')
        glove_vectors = {}
        with open('glove_vectors/glove.6B.50d.txt') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                word_vector = split_line[1:]
                word_vector = np.array([float(value) for value in word_vector])
                glove_vectors[word] = word_vector

    return glove_vectors


def tokenize_sentence(sentence):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(sentence)
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words


def clean_sentence(sentence):
    'Naive function to remove punctaion and special characters'
    return sentence.lower().replace('\n', ' ').replace('!', '').replace('?', '') # noqa


def sentences_to_vectors(sentences, word_embeddings):
    sentence_vectors = []
    for sentence in sentences:
        cleaned_sentence = clean_sentence(sentence)
        words = tokenize_sentence(cleaned_sentence)
        word_vectors = []
        for word in words:
            word_vector = word_embeddings.get(word, np.zeros(50,))
            word_vectors.append(word_vector)

        sentence_vector = sum(word_vectors) / len(word_vectors)
        sentence_vectors.append(sentence_vector)

    return sentence_vectors


def get_pagerank_scores(sentence_vectors):
    sim_mat = np.zeros([len(sentence_vectors), len(sentence_vectors)])

    for i, sentence in enumerate(sentence_vectors):
        for j, sentence in enumerate(sentence_vectors):
            sim_mat[i][j] = cosine_similarity(
                sentence_vectors[i].reshape(1, -1),
                sentence_vectors[j].reshape(1, -1)
            )[0, 0]

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    return scores


def summarize_file(filename, num_sentences=3):
    sentences = get_setences_from_file(filename)
    word_vectors = get_word_vectors()
    sentence_vectors = sentences_to_vectors(sentences, word_vectors)
    scores = get_pagerank_scores(sentence_vectors)

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_n_indices = [score[0] for score in sorted_scores[:num_sentences]]

    top_sentenes = [sentences[i] for i in top_n_indices]

    return ' '.join(top_sentenes)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    summary = summarize_file(args.filename)
    print(summary)
