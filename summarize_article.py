import os
import wget
import zipfile
import numpy as np
from nltk.tokenize import sent_tokenize

GLOVE_VECTORS_URL = 'http://nlp.stanford.edu/data/glove.6B.zip'


def get_setences_from_file(filename):
    with open(filename) as f:
        text = f.read().replace('\n', ' ')  # replace new lines with spaces

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


if __name__ == '__main__':
    sentences = get_setences_from_file('test-article.txt')
    print(sentences)
    word_vectors = get_word_vectors()
    print(len(word_vectors))
