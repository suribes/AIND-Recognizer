import warnings
from asl_data import SinglesData
import logging
import math
import arpa
import numpy as np
import copy
import operator
import re





def get_n_gram_score(sentence, n_gram_model, n_gram):

    logger = logging.getLogger('recognizer')
    # sentence.append(current_word)

    if len(sentence) > 0 and len(sentence) > n_gram:
        n_gram_len = n_gram
    else:
        n_gram_len = len(sentence)

    n_gram_sentence = " ".join(sentence[-n_gram_len:])
    # logger.info("Sentence {}".format(n_gram_sentence))

    try:
        n_gram_score = n_gram_model.log_s(n_gram_sentence)
    except:
        n_gram_score = n_gram_model.log_s("[UNKNOWN]")

    return n_gram_score

def get_n_gram_p(sentence, n_gram_model, n_gram):

    logger = logging.getLogger('recognizer')
    # sentence.append(current_word)

    if len(sentence) > 0 and len(sentence) > n_gram:
        n_gram_len = n_gram
    else:
        n_gram_len = len(sentence)

    n_gram_sentence = " ".join(sentence[-n_gram_len:])
    # logger.info("Sentence {}".format(n_gram_sentence))

    try:
        n_gram_p = n_gram_model.s(n_gram_sentence)
    except:
        n_gram_p = n_gram_model.s("[UNKNOWN]")
    # logger.info("Ngram score {}".format(n_gram_score))

    return n_gram_p




def recognize(models: dict, test_set: SinglesData, alpha = 0):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logger = logging.getLogger('recognizer')
    logger.info("Alpha {}".format(alpha))
    print("Alpha {}".format(alpha))

    probabilities = []
    guesses = []
    lm_models = arpa.loadf("devel-lm-M3.sri.lm")
    lm_model = lm_models[0]

    # TODO implement the recognizer
    logger.debug("Models {}".format(models))
    sequences = test_set.get_all_sequences()
    logger.debug("Sequences {}".format(sequences))
    logger.debug("Test words {}".format(test_set.wordlist))
    logger.debug("Sentences {}".format(test_set._load_sentence_word_indices()))

    sentence_start = []
    for k, v in test_set._load_sentence_word_indices().items():
        sentence_start.append(v[0])

    sentence = [""]
    # for test_word_index in range(test_set.num_items):
    for test_word_index, test_word in enumerate(test_set.wordlist):
        test_X, test_lenghts = test_set.get_item_Xlengths(test_word_index)
        logger.debug("test_X {}".format(test_X))
        logger.debug("test_lenghts {}".format(test_lenghts))

        word_probabilities = {}
        word_scores = {}
        word_ngram_scores = {}
        word_n_gram_p = {}
        best_score = float("-inf")
        best_p = 0
        guess = None

        if test_word_index not in sentence_start:
            test_sentence = copy.deepcopy(sentence)
        else:
            test_sentence = ["<s>"]

        for word, model in models.items():
            logger.debug("Model {}".format(model))

            cleaned_word = re.sub(r'\d+$', '', word)
            test_sentence.append(cleaned_word)

            n_gram_score = get_n_gram_score(test_sentence, n_gram_model = lm_model, n_gram = 3)

            try:
                # n_gram_score = get_n_gram_score(test_sentence, n_gram_model = lm_model, n_gram = 3)
                score = model.score(test_X, test_lenghts) + alpha * n_gram_score
                # n_gram_p = math.exp(score) * get_n_gram_p(test_sentence, n_gram_model = lm_model, n_gram = 3)
                n_gram_p = math.exp(score) * get_n_gram_p(test_sentence, n_gram_model = lm_model, n_gram = 3)
            except:
                score = float("-inf")
                # p = 0
                n_gram_p = get_n_gram_p(test_sentence, n_gram_model = lm_model, n_gram = 3)
            # n_gram_p = math.exp(score) * get_n_gram_p(test_sentence, n_gram_model = lm_model, n_gram = 3)

            word_probabilities.update({word: math.exp(score)})
            word_scores.update({word: score})
            word_ngram_scores.update({word: n_gram_score})
            word_n_gram_p.update({word: n_gram_p})
            if score > best_score:
                best_score = score
                guess = word

            if test_word_index not in sentence_start:
                test_sentence = copy.deepcopy(sentence)
            else:
                test_sentence = ["<s>"]
        guesses.append(guess)
        sentence.append(guess)
        probabilities.append(word_probabilities)
        logger.info("Test word {}".format(test_word))
        logger.info("Guess {}".format(guess))
        word_probabilities_ordered = sorted(word_probabilities.items(), key=operator.itemgetter(1), reverse = True)
        # logger.info("Probability {}".format(word_probabilities))
        logger.info("Probability {}".format(word_probabilities_ordered))
        word_scores_ordered = sorted(word_scores.items(), key=operator.itemgetter(1), reverse = True)
        logger.info("Scores {}".format(word_scores_ordered))
        word_ngram_scores_ordered = sorted(word_ngram_scores.items(), key=operator.itemgetter(1), reverse = True)
        logger.info("Ngram Scores {}".format(word_ngram_scores_ordered))
        word_ngram_p_ordered = sorted(word_n_gram_p.items(), key=operator.itemgetter(1), reverse = True)
        logger.info("Ngram P {}".format(word_ngram_p_ordered))

    return probabilities, guesses

if __name__ == "__main__":
    from  asl_test_recognizer import TestRecognize
    from utils import config_log

    config_log()
    logger = logging.getLogger('recognizer')
    test_model = TestRecognize()
    test_model.setUp()
    test_model.test_recognize_probabilities_interface()
    test_model.test_recognize_guesses_interface()
