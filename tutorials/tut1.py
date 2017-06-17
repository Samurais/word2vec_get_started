# https://radimrehurek.com/gensim/tut1.html
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora
documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]


'''
build texts
'''
# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
    for document in documents]

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1]
         for text in texts]
from pprint import pprint  # pretty-printer
pprint(texts)

'''
build dict
'''
dictionary = corpora.Dictionary(texts)
dictionary.save('../data/deerwester.dict')  # store the dictionary, for future reference
print(dictionary)
# the mapping between words and their ids:
print(dictionary.token2id)

'''
convert tokenized documents to vectors
'''
new_doc = "Human computer interaction computer"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)  # the word "interaction" does not appear in the dictionary and is ignored

'''
generate files
'''
class MyCorpus(object):
    def __iter__(self):
        for line in open('../data/mycorpus.txt'):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())

corpus = []
corpus_memory_friendly = MyCorpus()  # doesn't load the corpus into memory!
for vector in corpus_memory_friendly: # load one vector into memory at a time
    print(vector)
    corpus.append(vector)

corpora.MmCorpus.serialize('../data/deerwester.mm', corpus)
print("Saved ../data/deerwester.mm")
print('Done.')
