import glob
import sys
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import pdb
import ujson
import time
import math


# Create Co-occurrence tuple for wnd[-1] with all other wnd[i]
# Get the expansion term for a word.
def get_expan_terms(aword):
    return [rw for rw, fw in cfd[aword].most_common(TOP_N)]

def get_exapn_for_query(query_text):
    qt = nltk.word_tokenize(query_text)
    expans = nltk.FreqDist()
    for query_term in qt:
        expan_candidats = get_expan_terms(query_term)
        # Because the different query item may get the same expansion term.
        # We should sum the same expansion term scores for the different query items
        list_scores = {ex_term: cfd[query_term][ex_term] for ex_term in expan_candidats}
        # P(term_j | Q)
        #    =      lambda * P_ml(term_j | Query) +
        #             (1-lambda)* sum{ P( term_j | term_i) * P_ml( term_i | Query) }
        #    =      l * frequency_in_query(term_j)/length(Query) +
        #              (1-l)* sum_{i}{ score_term_term(term_i, term_j) * frequency_in_query(term_i)/length(Query) }
        # Here we do the sum part
        expans = expans + nltk.FreqDist(list_scores)
    return expans

# Calculate P( term_j | term_i )
# For the cooccurrence,
# term_i and term_j is the two terms in the corpus
#
#                                  #coocc(term_i, term_j)
# P( term_j | term_i )  =     -----------------------------
#                            sum_k  #coocc(term_i, term_k)
#
#                                  P( term_i, term_j)
#  PMI( term_i, term_j) =  log10 ---------------------------
#                                  P( term_i) P(term_j)
#
#                                  #coocc(term_i, term_j) x number_of_all_the_cooccurences
#                       =  log10 ---------------------------------------------------------------
#                                  sum_k  #coocc(term_i, term_k) x sum_k  #coocc(term_j, term_k)
#
#                                  #coocc(term_i, term_j) x N x TERM_DISTANCE x 2
#                       =  log10 ---------------------------------------------------------------
#                                  freq(term_i) x freq(term_j) x  (TERM_DISTANCE x 2) ^2
#
#                                   #coocc(term_i, term_j) x N
#                       =  log10 ---------------------------------------------------------------
#                                  freq(term_i) x freq(term_j) x  (TERM_DISTANCE x 2)
def score_term_in_term(term_j, term_i, N):
    global cfd
    if PMI_FLAG:
        pmi = math.log10(cfd[term_i][term_j]*N / (list_freq[term_i]*list_freq[term_j]*(TERM_DISTANCE*2)))
        r = pmi
    else:
        p_term_j_in_term_i = cfd[term_i][term_j] / (list_freq[term_i]*TERM_DISTANCE*2)
        r = p_term_j_in_term_i
    return r


# Indri va faire ça, on ne fait pas le calcul
# P(term_j | Q)
#    =      lambda * P_ml(term_j | Query) +
#             (1-lambda)* sum{ P( term_j | term_i) * P_ml( term_i | Query) }
#    =      l * frequency_in_query(term_j)/length(Query) +
#              (1-l)* sum_{i}{ score_term_term(term_i, term_j) * frequency_in_query(term_i)/length(Query) }
#
# def score_term_in_query(term_j, qt_list, l=0.5):
#     fd = nltk.FreqDist(qt_list)
#     # If term_j is not in the fd, fd[term_j] equals 0
#     r = l * fd[term_j] / len(qt_list) + \
#         (1-l) * sum([cfd[term_i][term_j] * fd[term_i]/len(qt_list) for term_i in qt_list])
#     return r

wnd = wnd[-WND_SIZE:] #push to list from last -WND_SIZE (here is -6)
def add_conditional_frequence_table(wnd):
    global cfd
    new_term = wnd[-1]
    for term in wnd[-WND_SIZE:-1]: # get the five terms before the new_term
            # get the frequent of two terms considered the order
            cfd[term][new_term] += 1
            cfd[new_term][term] += 1


# Read the cfd.json file
def reload_cfd_json(fname):
    global cfd
    cfd_list = ujson.load(open(fname))
    cfd = nltk.ConditionalFreqDist()
    for w in cfd_list:
        cfd[w] = nltk.FreqDist(cfd_list[w])
    return cfd

#/Users/jeanneluo/UdeM/IFT6255_RI/devoir1/AP/AP/*
#/Users/jeanneluo/UdeM/IFT6255_RI/devoir1/Code/ap_cfd{dist}{min}{top}{stop}{pmi}.json
def extract_cooccurence():
    global cfd, list_freq
    #if len(sys.argv) > 1:
        # Define the data path
    #    data_path = sys.argv[1]
    data_path = "/Users/jeanneluo/UdeM/IFT6255_RI/devoir1/AP/AP/AP880612"
    start_time = time.time()

    list_of_file = sorted(glob.glob(data_path))
    cfd = nltk.ConditionalFreqDist()
    list_freq = nltk.FreqDist()

    stop = set(stopwords.words('english'))
    if not STOP_FLAG:
        stop = []
    ps = PorterStemmer()

    for index, fname in enumerate(list_of_file):
        print("No.{} File: {}".format(index+1, fname))
        with open(fname, encoding='latin') as file:
            raw = file.read()
            # Extract all the <TEXT> field
            result = re.findall(r'<TEXT>(.*?)</TEXT>', raw, re.DOTALL)
            texts = ''.join(result)
            # Tokenize
            tokens = word_tokenize(texts)
            # Filter Tokens is alphabetical and keep the in lower case
            # Filter by stopwords
            #tokens_norm = [t.lower() for t in tokens if t.isalpha() and (t.lower() not in stop)]
            tokens_norm = ['app', 'mickey', 'mouse', 'key', 'rush', 'cell', 'pan', 'egg', 'mouse', 'key', 'rush','mouse', 'key', 'recall', 'relax', 'for', 'key', 'enya', 'song', 'panda', 'charger', 'key', 'cell', 'key']
            # Count the Frequency for each word
            list_freq += nltk.FreqDist(tokens_norm)

            # Tokes neighbors window
            #wnd = [''*WND_SIZE]
            wnd = []
            for t in tokens_norm:
                wnd.append(t)
                wnd = wnd[-WND_SIZE:] #push to list from last -WND_SIZE (here is -6)
                # Add to conditional frequency table
                add_conditional_frequence_table(wnd)

    print("Time1: {}".format(time.time() - start_time))

    cfd_filter = nltk.ConditionalFreqDist()
    # Filter the MIN_COOCC and Calculate the score

    # Calculate cfd.N()
    total_N = list_freq.N()
    for term_i in cfd:
        cfd_filter[term_i] = nltk.FreqDist({term_j: score_term_in_term(term_j, term_i, total_N)
                                            for term_j in cfd[term_i] if cfd[term_i][term_j] > MIN_COOCC})
        # Don't count the word itself as a relation
        if term_i in cfd[term_i]:
            cfd[term_i].pop(term_i)
    print("Time2: {}".format(time.time() - start_time))
    cfd_topn = nltk.ConditionalFreqDist()
    # Get the TOP N
    for w in cfd_filter:
        cfd_topn[w] = nltk.FreqDist(dict(cfd_filter[w].most_common(DOUBLE_TOP_N)))
    print("Time3: {}".format(time.time() - start_time))

    print("Time4: {}".format(time.time() - start_time))

    file_tag = {
        'dist': '_dist'+str(TERM_DISTANCE),
        'min': '_min'+str(MIN_COOCC),
        'top': '_top'+str(TOP_N),
        'stop': '_stp' if STOP_FLAG else '',
        'pmi': '_pmi' if PMI_FLAG else ''
    }

    ujson.dump(cfd_topn, open("/Users/jeanneluo/UdeM/IFT6255_RI/devoir1/Code/ap_cfd{dist}{min}{top}{stop}{pmi}.json".format(
        **file_tag), "w"), double_precision=3)

    print("Time5: {}".format(time.time() - start_time))
    #pdb.set_trace()
    return cfd_topn


# CONSTANTS
TERM_DISTANCE = 5
WND_SIZE = TERM_DISTANCE + 1
MIN_COOCC = 10
TOP_N = 10
DOUBLE_TOP_N = TOP_N * 2
PMI_FLAG = False
STOP_FLAG = True

# GLOBALS
cfd = nltk.ConditionalFreqDist()


if __name__ == "__main__":
    extract_cooccurence()
