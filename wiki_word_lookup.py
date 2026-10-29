import numpy as np
import argparse
from sklearn.preprocessing import normalize
from typing import override
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer


from WikiDataReader import *

#####################################################################
# Part 3:
# Define your new subclasses below!
#####################################################################

class SVDWordVectors(WikiWordVectors):

    def __init__(self):
        '''
        Create an object of SVDDocumentVectors containing:
            TfidfVectorizer and word_vectors (vh matrix, after running svds to the doc-term matrix)
        '''
        self.vectorizer = TfidfVectorizer(max_features=10000, max_df=0.70)
        self.word_vectors = None

    @override
    def train(self, documents):
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        u, s, vh = svds(A=doc_term_matrix, k=100)
        self.word_vectors = vh
        return u
    
    @override
    def get_word_index(self, word):
        pass 
    
    @override
    def get_vocabulary(self):
        pass
    
    @override
    def get_word_vector(self, word):
        pass


#####################################################################
# WikiWordLookup
#####################################################################

class WikiWordLookup():

    def __init__(self, articles, word_vector_model):
        '''
        Create a WikiWordLookup object to search for most ranked words from provided articles
        and training with a given word_vector_model to obtain its wiki_word_vectors.
        '''
        self.articles = articles
        self.word_vector_model = word_vector_model
        self.wiki_word_vectors = None

    def train_vector_model(self):
        '''
        Train the instance's word_vector_model with the articles' abstracts
        and update wiki_word_vectors
        '''
        print("Training vector model. Please wait, this may take a while...")
        self.wiki_word_vectors = self.word_vector_model.train(self.articles.get_abstracts())
        self.wiki_word_vectors = normalize(self.wiki_word_vectors)
        print("Done!")

    def compute_rankings(self, input_word):
        '''
        For a given input_word, obtain the word vector from the word_vector_model and 
        compute the similarity score rankings between this word vector and all other 
        word vectors in wiki_word_vectors. 
        
        This method can only be called after train_vector_model() has been called.

        Returns a tuple containing:
            - similarity scores array
            - rankings of the similarity scores from most similar to least similar 
              (besides the similarity of the input_word)
        
        If there is no word_vector, then input_word is not in vocabulary, so return (None, None)
        '''
        if self.wiki_word_vectors is None:
            raise RuntimeError("Please run train_vector_model first!")

        word_vector = normalize(self.word_vector_model.get_word_vector(input_word))
        if word_vector is None:
            return None, None

        # TODO: REPLACE the following line with code that computes the
        # cosine similarity between the vector for input_word and all
        # other word vectors in the self.wiki_word_vectors matrix, and
        # saves the similarity scores to a list or numpy array called 
        # sims, indexed in the same order as self.wiki_word_vectors.
        sims = []

        return sims, np.argsort(sims)[::-1][1:]

    def run_search(self, input_word, n_results=10):
        '''
        For a given input_word, print out n_results (defaulted at 10) results of the most matched words
        with its corresponding similarity ranking to the input-word.
        '''
        input_word = input_word.lower()
        print(f"Top {n_results} most similar words to {input_word}:")
        sims, ranking = self.compute_rankings(input_word)
        if ranking is None:
            print(f"'{input_word}' not found in the vocabulary!")
        else:
            for rank, i in enumerate(ranking[:n_results]):
                print(f"{rank+1}. {self.word_vector_model.get_vocabulary()[i]} (similarity={sims[i]})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str,
                        help="Directory containing the Wikipedia corpus files")
    parser.add_argument("-w", "--word", type=str, metavar="W", required=True,
                        help="The word to look up")
    parser.add_argument("-n", "--num_articles", type=int, metavar="N", default=None,
                        help="Only keep the first N articles parsed from the corpus files")
    
    args = parser.parse_args()

    articles = WikiArticles(args.data_dir, args.num_articles)
    # TODO replace the following line with an initialization of the appropriate WikiWordVectors subclass
    word_vector_model = None

    word_lookup = WikiWordLookup(articles, word_vector_model)
    word_lookup.train_vector_model()
    word_lookup.run_search(args.word)

if __name__ == "__main__":
    main()
