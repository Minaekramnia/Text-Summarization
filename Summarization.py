# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:38:30 2019

@author: wb550776
"""
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt') # one time execution
import re
import docx
from nltk.tokenize import sent_tokenize
#pip uninstall docx
#pip install python-docx
from docx import Document
document = Document()
document = Document('C:/Users/wb550776/Downloads/Using ML in Evaluative Synthesis May 31 2019.docx')

#Open the article
def read_article(filename):
    file = open(r'C:/Users/wb550776/Downloads/10144-Ghana-ELAC EvNote.docx', encoding='utf-8')
    #file2 = open(r'C:/Users/wb550776/Downloads/README.txt')
    filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences = []

### IF the file is .csv format
df = pd.read_csv(r"C:\Users\wb550776\Documents\IEG_Sumerization\tennis_articles_v4.csv")
doc = docx.Document(r'C:/Users/wb550776/Downloads/Using ML in Evaluative Synthesis May 31 2019.docx')
text = []
for i in doc.paragraphs:
    text.append(i.text)

#Split Text in to Sentences
sentences = []
for s in text:
  sentences.append(sent_tokenize(s))

sentences = [y for x in sentences for y in x] # flatten list

#for s in df['article_text']:
#  sentences.append(sent_tokenize(s))

#!wget http://nlp.stanford.edu/data/glove.6B.zip
#!unzip glove*.zip

# Extract the word vectors
#cd C:\Users\wb550776\Documents\Projects\Summarization
word_embeddings = {}

f = open(r'C:\Users\wb550776\Documents\IEG_Sumerization\glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

len(word_embeddings)
#We now have word vectors for 400,000 different terms stored in the dictionary – ‘word_embeddings’.

###################
#Text Preprocessing
# remove punctuations, numbers and special characters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# make alphabets lowercase
clean_sentences = [s.lower() for s in clean_sentences]

#Get rid of Stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

#define a function to remove the stopwords from the dataset:
# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new


# remove stopwords from the sentences
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]


####################################
##Vector Representation of Sentences

# Extract word vectors
word_embeddings = {}
f = open(r'C:\Users\wb550776\Documents\IEG_Sumerization\glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

##
#Now, let’s create vectors for our sentences. We will first fetch vectors 
#(each of size 100 elements) for the constituent words in a sentence and then 
#take mean/average of those vectors to arrive at a consolidated vector for the sentence.

sentence_vectors = []
for i in clean_sentences:
  if len(i) != 0:
    v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
  else:
    v = np.zeros((100,))
  sentence_vectors.append(v)

##############################
###Step 3: Rank Sentences in Similarity Matrix 

#Similarity matrix Preparation
sim_mat = np.zeros([len(sentences), len(sentences)])

#We will use Cosine Similarity to compute the similarity between a pair of sentences

from sklearn.metrics.pairwise import cosine_similarity

for i in range(len(sentences)):
  for j in range(len(sentences)):
    if i != j:
      sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
############################
## Step 4 - Sort the rank and pick top sentences
#Applying PageRank Algorithm
import networkx as nx
nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)
########################
##Summary Extraction
ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
# Extract top 10 sentences as the summary
for i in range(2):
  print(ranked_sentences[i][1])


### If the file is a text .txt format
file = 'input.txt'
file = open(r'C:/Users/wb550776/Downloads/README.txt')
file = open(r'C:/Users/wb550776/Downloads/Using ML in Evaluative Synthesis May 31 2019.docx')
#file = open(file , 'r')
text = file.read()
tokenized_sentence = sent_tokenize(text)
sentences = tokenized_sentence

text = remove_special_characters(str(text))
text = re.sub(r'\d+', '', text)
tokenized_words_with_stopwords = word_tokenize(text)
tokenized_words = [word for word in tokenized_words_with_stopwords if word not in Stopwords]
tokenized_words = [word for word in tokenized_words if len(word) > 1]
tokenized_words = [word.lower() for word in tokenized_words]

# output to excel
writer = pd.ExcelWriter(os.path.join('tables', 'document_prediction.xlsx'))
df.to_excel(writer, sheet_name='project_prediction')
result_list[0].to_excel(writer, sheet_name='category1_salient_prediction')
# df_result_topic.to_excel(writer, sheet_name='salient_prediction_by_topic')
writer.save()

#The output in excel
#go over different projects, number of project. output is number: top 1, top 5. 
#readme. 
#put things in the left, get the outputs. 
#README. Try on his computer. 
#Then different algorithms. accuracy? 