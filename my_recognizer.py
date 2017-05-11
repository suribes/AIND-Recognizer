import warnings
from asl_data import SinglesData
import logging
import math
import arpa
import numpy as np



def get_n_gram_score(current_word, sentence, n_gram_model, n_gram):

    n_gram_sentence = " ".join(sentence[-n_gram:])

    try:
        score = n_gram_model.log_p(n_gram_sentence)
        n_gram_score = score
    except:
        n_gram_score = 0

    return n_gram_score

def get_n_gram_p(current_word, sentence, n_gram_model, n_gram):

    n_gram_sentence = " ".join(sentence[-n_gram:])

    try:
        p = n_gram_model.p(n_gram_sentence)
        n_gram_p = p
    except:
        n_gram_p = 0

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
    logger.info("Sentences {}".format(test_set._load_sentence_word_indices()))
    for test_word_index in range(test_set.num_items):
    # for test_word_index, test_word in enumerate(test_set.wordlist):
        test_X, test_lenghts = test_set.get_item_Xlengths(test_word_index)
        logger.debug("test_X {}".format(test_X))
        logger.debug("test_lenghts {}".format(test_lenghts))

        word_probabilities = {}
        best_score = float("-inf")
        best_p = 0
        guess = None
        sentence = [""]

        diffs = []
        for word, model in models.items():
            logger.debug("Model {}".format(model))
            sentence.append(word)
            try:
                # L(GIVE) = log(exp(L(GIVE)) + exp(L(GIVE1))
                n_gram_score = get_n_gram_score(word, sentence, n_gram_model = lm_model, n_gram = 2)
                # n_gram_p = get_n_gram_p(word, sentence, n_gram_model = lm_model, n_gram = 3)
                # score = model.score(test_X, test_lenghts)
                score = model.score(test_X, test_lenghts) + alpha * n_gram_score
                diffs.append(model.score(test_X, test_lenghts) - n_gram_score)
                # score = math.log(math.exp(model.score(test_X, test_lenghts)) + math.exp(n_gram_score))
                # p = math.exp(model.score(test_X, test_lenghts)) + n_gram_p / alpha
                # print("Score: {}".format(score))
                # logger.info("Score: {}".format(score))
                # logger.info("NGram score: {}".format(n_gram_score))
                # logger.info("Score: {}".format(p))
            except:
                score = float("-inf")
                # p = 0
            # probability.update({word: p})
            word_probabilities.update({word: math.exp(score)})
            if score > best_score:
                best_score = score
                guess = word
            # if p > best_p:
            #     best_p = p
            #     guess = word
        # logger.info("Diff average {}".format(sum(diffs) / float(len(diffs))))
        # print("Diff average {}".format(sum(diffs) / float(len(diffs))))
        guesses.append(guess)
        probabilities.append(word_probabilities)
        # logger.debug("Test word {}".format(test_word))
        logger.info("Guess {}".format(guess))
        logger.info("Probability {}".format(word_probabilities))


    # logger.info("Probabilities {}".format(probabilities))
    # logger.info("Guesses {}".format(guesses))
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
