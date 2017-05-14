import warnings
from asl_data import SinglesData
import logging
import math
import arpa
import numpy as np
import copy
import operator
import re
import pandas as pd



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


def get_sentence_score(sentence_indexes, models, test_set, probabilities, guesses, alpha_start = 0, alpha_previous = 0, alpha_transition = 0):
    logger = logging.getLogger('recognizer')

    top_best = 3

    lm_models = arpa.loadf("ukn.3.lm")
    lm = lm_models[0]

    emission_scores = get_emission_scores(sentence_indexes, models, test_set)
    guess = list(emission_scores.idxmax(axis = 0))
    guesses.extend(guess)

    word_probabilities = [v for k, v in emission_scores.to_dict().items()]
    probabilities.append(word_probabilities)
    logger.info("Guess {}".format(guess))
    logger.debug("Probability {}".format(word_probabilities))

    # # Create scores data frame
    # columns_list = ["scores_{}".format(i - sentence_offset) for i in (sentence_indexes)]
    # scores = pd.DataFrame(columns= columns_list)

    # # Iterate the observed sentence
    # for test_word_index in sentence_indexes:
    #     word_probabilities = {}

    #     # Get the observations
    #     word_sentence_index = test_word_index - sentence_offset
    #     score_column = "scores_{}".format(word_sentence_index)
    #     logger.debug("Score column {}".format(score_column))
    #     test_X, test_lenghts = test_set.get_item_Xlengths(test_word_index)
    #     if word_sentence_index > 0:
    #         score_column_previous = "scores_{}".format(word_sentence_index -1)
    #         best_previous_words = scores.sort_values(by=score_column_previous, ascending = False)[0:top_best].index
    #         logger.debug("Best previous words {}".format(best_previous_words))

    #     for word, model in models.items():
    #         # Build sentence
    #         clean_word = re.sub(r'\d+$', '', word)

    #         # Calculate emission score
    #         try:
    #             emission_score = model.score(test_X, test_lenghts)
    #         except:
    #             emission_score = float("-inf")

    #         # Calculate best score = emission score + transition score + best previous score
    #         if word_sentence_index > 0:
    #             best_middle_score = float("-inf")
    #             for previous_word in best_previous_words:
    #                 ngram_middle_sentence = [previous_word]
    #                 ngram_middle_sentence.append(clean_word)
    #                 transition_score = get_n_gram_score(ngram_middle_sentence, n_gram_model = lm, n_gram = 3)
    #                 previous_best_score = scores.get_value(previous_word, score_column_previous)
    #                 score = alpha_transition * transition_score + alpha_previous * previous_best_score

    #                 if score > best_middle_score:
    #                     best_middle_score = score
    #             best_score = emission_score +  best_middle_score
    #         else:
    #             ngram_sentence = ["<s>"]
    #             ngram_sentence.append(clean_word)
    #             transition_score = get_n_gram_score(ngram_sentence, n_gram_model = lm, n_gram = 3)
    #             best_score = emission_score + alpha_start * transition_score

    #         # Aggregate score
    #         try:
    #             old_score = scores.get_value(clean_word, score_column)
    #         except:
    #             old_score = 0
    #         best_score += old_score

    #         scores.set_value(clean_word, score_column, best_score)
    #         logger.debug("Best score {}".format(best_score))
    #         word_probabilities.update({word: math.exp(emission_score)})

        # guess = scores.idxmax(axis = 0)[score_column]
        # logger.info("Best scores {}".format(scores))
        # guesses.append(guess)
        # probabilities.append(word_probabilities)
        # logger.info("Guess {}".format(guess))
        # logger.debug("Probability {}".format(word_probabilities))
    return emission_scores


def get_emission_scores(sentence_indexes, models, test_set):
    logger = logging.getLogger('recognizer')

    sentence_offset = sentence_indexes[0]

    # Create scores data frame
    columns_list = ["scores_{}".format(i - sentence_offset) for i in (sentence_indexes)]
    scores = pd.DataFrame(columns= columns_list)

    for test_word_index in sentence_indexes:
        test_X, test_lenghts = test_set.get_item_Xlengths(test_word_index)
        score_column = "scores_{}".format(test_word_index - sentence_offset)
        logger.debug("Score column {}".format(score_column))

        for word, model in models.items():
            # Build sentence
            clean_word = re.sub(r'\d+$', '', word)
            logger.debug("Clean word {}".format(clean_word))

            # Calculate emission score
            try:
                emission_score = model.score(test_X, test_lenghts)
            except:
                emission_score = float("-inf")

            # Aggregate score
            try:
                score = scores.get_value(clean_word, score_column)
            except:
                score = 0
            logger.debug("Emission score {}".format(emission_score))
            score += emission_score
            logger.debug("Score {}".format(score))
            scores.set_value(clean_word, score_column, score)
        scores.fillna(0, inplace=True)
    return scores



def recognize(models: dict, test_set: SinglesData, alpha_start = 0, alpha_previous = 0, alpha_transition = 0):
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
    logger.info("Alpha {}".format(alpha_start))
    print("Alpha start {}".format(alpha_start))
    print("Alpha previous {}".format(alpha_previous))
    print("Alpha transition {}".format(alpha_transition))

    probabilities = []
    guesses = []


    # TODO implement the recognizer
    # logger.debug("Models {}".format(models))
    # sequences = test_set.get_all_sequences()
    # logger.debug("Sequences {}".format(sequences))
    # logger.debug("Test words {}".format(test_set.wordlist))
    # logger.debug("Sentences {}".format(test_set._load_sentence_word_indices()))

    sentences = test_set._load_sentence_word_indices()
    sentences_indexes = []
    for k, v in sentences.items():
        sentences_indexes.append(v)
    sentences_indexes

    # for test_word_index, test_word in enumerate(test_set.wordlist):
    # for test_word_index in range(test_set.num_items):
    for sentence_indexes in sentences_indexes:
        sentence_score = get_sentence_score(sentence_indexes, models, test_set, probabilities, guesses, alpha_start, alpha_previous)
        logger.info("Sentence indexes {}".format(sentence_indexes))
        logger.debug("Sentence score {}".format(sentence_score))
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

