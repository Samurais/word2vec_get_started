import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora, models, similarities

if (os.path.exists("../data/deerwester.dict")):
     dictionary = corpora.Dictionary.load('../data/deerwester.dict')
     corpus = corpora.MmCorpus('../data/deerwester.mm')
     print("Used files generated from first tutorial")
else:
    print("Please run first tutorial to generate data set")

'''
initialize a TF-IDF model
'''
tfidf = models.TfidfModel(corpus) 

'''
transform vectors
'''
doc_bow = [(1, 1), (2, 1)]
# corpus is the collection
# doc_bow is just a single document
# get the tfidf value
# http://blog.chatbot.io/development/2017/06/14/tf-idf/
print(tfidf[doc_bow])


'''
apply a transformation to a whole corpus
'''
corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)

'''
chain transformations
https://en.wikipedia.org/wiki/Latent_semantic_indexing
'''
# num_topics, for the toy corpus above we used only 2 latent dimensions, but on real corpora, target 
# dimensionality of 200–500 is recommended as a “golden standard”
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
lsi.print_topics(2)

'''
save and load model
'''
lsi.save('../data/model.lsi') # same for tfidf, lda, ...
lsi = models.LsiModel.load('../data/model.lsi')