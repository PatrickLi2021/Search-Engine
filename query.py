from queue import PriorityQueue
import sys
import file_io
import sys
from pickle import STOP
from file_io import write_title_file
from file_io import write_words_file
from file_io import write_docs_file
import xml.etree.ElementTree as et
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words('english'))
nltk_test = PorterStemmer()

def main():
    """
    This function reads in the number of arguments and determines whether to 
    factor PageRank into the final rankings. It then passes data into the REPL
    to rank and print out the 10 most relevant pages (or if there are less than 
    10 total pages, ranks the pages in the corpus).

    Parameters:
    None

    Returns:
    None
    """

    # Checks the number of arguments the user passes in and whether PageRank is
    # to be included
    if len(sys.argv) - 1 != 3 and len(sys.argv) - 1 != 4:
        print("Invalid number of arguments")
        sys.exit
    elif sys.argv[1] == "--pagerank":
        repl(True, sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        repl(False, sys.argv[1], sys.argv[2], sys.argv[3])

def stem_stop(list_of_words: list):
    """
    This function removes all stop words from a list of strings and stems
    each word (if applicable) in the list

    Parameters:
    list_of_words: A list of words that has been tokenized already

    Returns:
    A list of strings in which each string in the list has been stemmed
    and all the stop words have been removed
    """

    for i in range(len(list_of_words)-1, -1, -1):
        if list_of_words[i] in STOP_WORDS:
            list_of_words.remove(list_of_words[i])
        else:
            list_of_words[i] = nltk_test.stem(list_of_words[i])
    return list_of_words

def repl(use_pagerank: bool, title_index: str, doc_index: str, word_index: str):
    """
    This function prompts the user for a query and answers the query by scoring 
    its terms against every document. Lastly, it computes the final document 
    rankings and prompts the user for additional input

    Parameters:
    use_pagerank: A boolean, true, if the user wants to account for PageRank, 
    and false otherwise
    title_index: The filepath to the file that contains the IDs and titles
    doc_index: The filepath to the docs file
    word_index: The filepath that the words_to_doc_frequency dictionary was 
    written to

    Returns:
    None
    """

    # Initializes the ids_to_titles, ids_to_page_ranks, and term relevance
    # dictionaries to be empty so that they can be read in via file.io
    ids_to_titles = {}
    ids_to_page_ranks = {}
    words_to_ids_to_term_relevance = {}
    file_io.read_title_file(title_index, ids_to_titles)
    file_io.read_docs_file(doc_index, ids_to_page_ranks)
    file_io.read_words_file(word_index, words_to_ids_to_term_relevance)
    print(words_to_ids_to_term_relevance)

    query = input("Please enter query: ").lower().split()
    doc_ranking = PriorityQueue()
    while query != [":quit"]:
        stem_stop_query = stem_stop(query)

        # Runs the inputted query against every document in the corpus
        for document in ids_to_page_ranks.keys():

            # Scores query against documents using PageRank
            if use_pagerank:
                score = score_terms_page_rank(stem_stop_query, document,
                                              ids_to_page_ranks,
                                              words_to_ids_to_term_relevance)
                # Makes score negative because priority queue stores from
                # least to greatest
                if (score != 0):
                    doc_ranking.put((-score, document))

            # Scores query against documents without PageRank
            else:
                score = score_terms(stem_stop_query, document,
                                    words_to_ids_to_term_relevance)
                # Makes score negative because priority queue stores from
                # least to greatest
                if (score != 0):
                    doc_ranking.put((-score, document))

        # Determines the final ranking of documents and prompts the user to
        # enter another query into the terminal

        find_final_rankings(doc_ranking, ids_to_titles)
        query = input("Please enter query: ").lower().split()
        doc_ranking = PriorityQueue()

def score_terms(list_of_words: list, id: int,
                words_to_ids_to_term_relevance: dict):
    """
    This function computes the relevance scores for the query against a 
    particular document in the corpus without taking into account PageRank

    Parameters:
    list_of_words: A list of string(s) representing word(s) in the query
    id: The ID of the document that the query is being scored against as an int
    words_to_ids_to_term_relevance: A dictionary that maps words to IDs to 
    their corresponding term relevances

    Returns:
    The score of a particular document based only on term relevance
    """
    score = 0
    for word in list_of_words:
        if word in words_to_ids_to_term_relevance.keys() and \
           id in words_to_ids_to_term_relevance[word].keys():
            score = score + words_to_ids_to_term_relevance[word][id]
    return score

def score_terms_page_rank(list_of_words: list, id: int,
                          ids_to_page_ranks: dict,
                          words_to_ids_to_term_relevance: dict):
    """
    This function scores a particular document against the query taking into 
    account both term relevance and PageRank

    Parameters:
    list_of_words: A list of string(s) representing word(s) in the query
    id: The ID of the document that the query is being scored against as an int
    ids_to_page_ranks: A dictionary that maps a document's ID to its 
    corresponding PageRank value
    words_to_ids_to_term_relevance: A dictionary that maps words to IDs to 
    their corresponding term relevances

    Returns:
    The score of a particular document based only on term relevance
    """

    sum = 0
    for word in list_of_words:
        if word in words_to_ids_to_term_relevance.keys() and \
                id in words_to_ids_to_term_relevance[word].keys():
            sum = sum + words_to_ids_to_term_relevance[word][id]
    return sum * ids_to_page_ranks[id]

def find_final_rankings(doc_ranking: PriorityQueue, ids_to_title: dict):
    """
    This function prints out the final rankings of documents in the order of 
    highest relevance and/or authority to lowest relevance and/or authority
    unless all scores are 0 meaning no search results would be found in the 
    corpus.

    Parameters:
    doc_ranking: A priority queue that stores each page's combined relevance 
    and/or authority metric in order from lowest to highest
    ids_to_title: A dictionary that maps a document ID to its corresponding 
    title

    Returns:
    None
    """
    if doc_ranking.empty():
        print("No search results were found")
    for i in range(1, 11):
        if doc_ranking.empty():
            break
        else:
            score_doc_pair = doc_ranking.get()
            print(ids_to_title[score_doc_pair[1]])

try:
    main()
except FileNotFoundError as e:
    print("File was not found.")
except IOError as e:
    print("Error reading in file.")