from nltk.tokenize import sent_tokenize


def get_setences_from_file(filename):
    with open(filename) as f:
        text = f.read().replace('\n', ' ')  # replace new lines with spaces

    return sent_tokenize(text)


if __name__ == '__main__':
    sentences = get_setences_from_file('test-article.txt')
    print(sentences)
