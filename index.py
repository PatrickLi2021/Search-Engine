import math
import sys
from pickle import STOP
from file_io import write_title_file
from file_io import write_words_file
from file_io import write_docs_file
import re
import xml.etree.ElementTree as et
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words('english'))
nltk_test = PorterStemmer()

class Index:

    """
    This is a class that contains code for processing an XML document into a
    list of terms, determining the relevance between terms and documents, and
    determining the authority/rank of each document.
    """

    def __init__(self, xml_filepath: str, titles_filepath: str,
                 docs_filepath: str, words_filepath: str):
        """ 
        This is the constructor for the Index class. It initializes the XML,
        titles, docs, words, and roots fields, and calls the parse_xml function,
        and it will call methods in file_io to write the contents of the
        dictionaries there.

        Parameters:
        xml_filepath: the string representing the filepath to the xml file
        titles_filepath: the string representing the filepath to the titles file
        docs_filepath: the string representing the filepath to the docs file
        words_filepath: the string representing the filepath to the words file

        Returns:
        None
        """

        self.title_to_id = {}
        self.id_to_title = {}
        self.page_to_links = {}
        self.words_to_ids_to_term_relevance = {}
        self.ids_to_page_ranks = {}
        self.weights = {}

        try:
            self.root = et.parse(xml_filepath).getroot()
            self.populate_id_and_title_dicts()
            self.parse_xml()
            write_title_file(titles_filepath, self.id_to_title)
            write_words_file(words_filepath,
                             self.words_to_ids_to_term_relevance)
            write_docs_file(docs_filepath, self.ids_to_page_ranks)
        except FileNotFoundError as e:
            print("File was not found")
        except IOError as e:
            print("Error reading in file.")

    def populate_id_and_title_dicts(self):
        """
        This function populates both the id_to_title and title_to_id
        dictionaries by looping through each page in the corpus

        Parameters:
        None

        Returns:
        None
        """

        # for loop to populate ID to title and title to ID dictionaries
        all_pages = self.root.findall("page")
        for page in all_pages:
            # extracts the title and ID from the page and adds it to dictionary
            title = page.find('title').text.strip()
            id = page.find('id').text.strip()
            self.id_to_title[id] = title
            self.title_to_id[title] = id

    def parse_xml(self):
        """
        This function populates both the title to ID and ID to title
        dictionaries and will also call the tokenize, stem, and remove stop
        word functions for all text in each page in the corpus. This function
        will also call the term relevance and page rank methods to calculate
        those values.

        Parameters:
        None

        Returns:
        None
        """

        # for loop to tokenize, remove stop words, and stem text on each page
        all_pages = self.root.findall("page")
        for page in all_pages:
            title = page.find('title').text.strip()
            id = page.find('id').text.strip()
            self.page_to_links[id] = []
            no_links = True

            if page.find('text').text == None:
                page_text = title
            else:
                page_text = title + " " + page.find('text').text.strip()
            n_regex = \
                '''\[\[[^\[]+?\]\]|[a-zA-Z0-9]+'[a-zA-Z0-9]+|[a-zA-Z0-9]+'''

            # Initializes a list of all words in the text of a page
            words = re.findall(n_regex, page_text)

            # Goes through each link and only takes the right part of pipe
            for i in range(len(words) - 1, -1, -1):
                word = words[i]
                if (word[0:2] == "[[" and "|" in word):
                    # Getting text to the right of pipe to use for tokenizing
                    right_of_pipe = word.split("|")[1]
                    words.pop(i)
                    words = words + \
                        re.findall(
                            n_regex, right_of_pipe[0:len(right_of_pipe) - 2])

                    # Getting text to the left of pipe to populate page_to_links
                    # dictionary
                    left_of_pipe = word.split("|")[0][2:]
                    if (left_of_pipe in self.title_to_id and left_of_pipe not
                            in self.page_to_links[id] and left_of_pipe !=
                            self.id_to_title[id]):
                        # checks if the link is in corpus and doesn't add if
                        # it's already in there to avoid duplicates
                        self.page_to_links[id].append(left_of_pipe)
                        no_links = False

                elif (word[0:2] == "[["):
                    words.pop(i)
                    words = words + re.findall(n_regex, word[2: len(word) - 2])
                    # Populating the page_to_links dict
                    if (word[2: len(word) - 2] in self.title_to_id and
                            word[2: len(word) - 2] not in
                            self.page_to_links[id] and word[2: len(word) - 2]
                            != self.id_to_title[id]):
                        self.page_to_links[id].append(word[2: len(word) - 2])
                        no_links = False

            # If the page contains no links, page_to_links is populated with
            # all other pages in corpus except itself
            if (no_links == True):
                self.page_to_links[id] = [title for title in
                                          self.title_to_id.keys()
                                          if title != self.id_to_title[id]]
            words = self.stem_stop(words)
            self.calculate_term_frequency(words, id)
        self.calculate_term_relevance()
        self.page_rank()

    def stem_stop(self, list_of_words: list):
        """
        This function removes all stop words from a list of strings and stems
        each word (if applicable) in the list

        Parameters:
        list_of_words: A list of words that has been tokenized already

        Returns:
        A list of strings in which each string in the list has been stemmed
        and all the stop words have been removed
        """
        list_of_words = [x.lower() for x in list_of_words]
        for i in range(len(list_of_words)-1, -1, -1):
            if list_of_words[i] in STOP_WORDS:
                list_of_words.remove(list_of_words[i])
            else:
                list_of_words[i] = nltk_test.stem(list_of_words[i])
        return list_of_words

    def calculate_term_frequency(self, list_of_words: list, id: int):
        """
        This function populates the words_to_ids_to_term_relevance dictionary
        by finding the counts of each word in a document and then looping
        through and dividing each entry by aj.

        Parameters:
        list_of_words: A list of strings containing all the words in a 
        particular document
        id: an int representing the page id of the list_of_words

        Returns:
        None
        """

        # Initializes the max occurrences of a term in a page to be 0
        # (will be updated later)
        aj = 0

        # populate the self.ids_to_term_relevance dict at a word with the term
        # counts for the current page
        for word in list_of_words:
            if word in self.words_to_ids_to_term_relevance:
                if id in self.words_to_ids_to_term_relevance[word]:
                    count = self.words_to_ids_to_term_relevance[word][id] + 1
                    self.words_to_ids_to_term_relevance[word][id] = count
                else:
                    count = 1
                    self.words_to_ids_to_term_relevance[word][id] = 1

            else:
                count = 1
                self.words_to_ids_to_term_relevance[word] = {id: count}

            if count > aj:
                aj = count

        # change the term counts in self.words_to_ids_to_term_relevance to term
        # frequency by dividing all by aj
        for word in list(set(list_of_words)):
            for current_pg_id in self.words_to_ids_to_term_relevance[word]:
                if id == current_pg_id:
                    self.words_to_ids_to_term_relevance[word][id] = \
                        self.words_to_ids_to_term_relevance[word][id]/aj

    def calculate_term_relevance(self):
        """
        This function populates the words_to_ids_to_term_relevance dictionary
        by looping through the keys in the words_to_ids_to_term_relevance 
        dictionary and calculating the term relevance for a particular word

        Parameters:
        None

        Returns:
        None
        """

        num_of_docs = len(self.id_to_title)
        for word in self.words_to_ids_to_term_relevance.keys():
            for id in self.words_to_ids_to_term_relevance[word].keys():
                if id in self.words_to_ids_to_term_relevance[word]:
                    self.words_to_ids_to_term_relevance[word][id] = \
                        self.words_to_ids_to_term_relevance[word][id] * \
                        math.log((
                            num_of_docs /
                            len(self.words_to_ids_to_term_relevance[word])), 10)

    def euclidean_distance(self, prev_rankings: dict, current_rankings: dict):
        """
        This function calculates the Euclidean distance between the 2 ranking 
        vectors and returns the Euclidean distance.

        Parameters:
        prev_rankings: A dictionary that represents the previous iteration of 
        rankings and maps from doc ids to rankings

        current_rankings: A dictionary that represents the current iteration of 
        rankings for the pages and maps from doc ids to rankings

        Returns:
        float representing the Euclidean distance between the two vectors
        """

        sum = 0
        for id in prev_rankings.keys():
            sum = sum + (current_rankings[id] - prev_rankings[id]) ** 2
        return sum ** 0.5

    def find_weight(self, len_of_pages: int):
        """
        This function returns the weight of a page by using a formula that 
        takes into account the total number of pages and the number of unique 
        pages that link to the page

        Parameters:
        len_of_pages: The total number of pages in the corpus represented by an
        int

        Returns:
        The weight of a particular page represented as a float
        """

        for k in self.id_to_title.keys():
            # Filtered list of pages with duplicate pages removed
            links_lst = self.page_to_links[k]
            self.weights[k] = {}
            for j in self.id_to_title.keys():
                # If k has a link to j
                if j != k and self.id_to_title[j] in links_lst:
                    self.weights[k][j] = 0.15 / \
                        len_of_pages + 0.85/len(links_lst)

                # Otherwise (if j equals k or if k does NOT have a link to j)
                else:
                    self.weights[k][j] = 0.15/len_of_pages

    def page_rank(self):
        """
        This function calculates the page rankings of all the pages in the 
        corpus and stores them in the current rankings dictionary

        Parameters:
        pages: A list that contains all the pages in the corpus

        Returns:
        None
        """

        self.find_weight(len(self.title_to_id))

        # Initializes every rank in previous rankings dictionary to be 0
        prev_ids_to_page_ranks = {}
        prev_ids_to_page_ranks = prev_ids_to_page_ranks.fromkeys(
            self.id_to_title.keys(), 0)

        # Initializes every ranking in current rankings to be 1/n
        self.ids_to_page_ranks = {}
        self.ids_to_page_ranks = self.ids_to_page_ranks.fromkeys(
            self.id_to_title.keys(), 1/len(self.id_to_title))

        # Loop that continues to run as long as distance between 2 iterations
        # is above threshold
        while (self.euclidean_distance(prev_ids_to_page_ranks,
                                       self.ids_to_page_ranks) > 0.001):
            prev_ids_to_page_ranks = self.ids_to_page_ranks.copy()
            for j in self.id_to_title.keys():
                self.ids_to_page_ranks[j] = 0
                for k in self.id_to_title.keys():
                    self.ids_to_page_ranks[j] = self.ids_to_page_ranks[j] + \
                        self.weights[k][j] * prev_ids_to_page_ranks[k]
            sum = 0
            for i1 in self.ids_to_page_ranks.keys():
                sum = sum + self.ids_to_page_ranks[i1]

if __name__ == "__main__":
    i = Index(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])