import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora, models, similarities

'''
load previous dict and corpus
'''
dictionary = corpora.Dictionary.load('../data/deerwester.dict')
corpus = corpora.MmCorpus('../data/deerwester.mm') # comes from the first tutorial, "From strings to vectors"
print(corpus)

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)


'''
sort our nine corpus documents in decreasing order of relevance to this query
'''
doc = "Human computer interaction"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow] # convert the query to LSI space
print(vec_lsi)

'''
initializing query structures
'''
index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and index it
index.save('../data/deerwester.index')
index = similarities.MatrixSimilarity.load('../data/deerwester.index')
sims = index[vec_lsi] # perform a similarity query against the corpus
print(list(enumerate(sims)))

sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims) # print sorted (document number, similarity score) 2-tuples