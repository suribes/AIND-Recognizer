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


def get_sentence_score(sentence_indexes, models, test_set, probabilities, guesses, alpha_start, alpha_transition):
    logger = logging.getLogger('recognizer')
    # print("Alpha start {}".format(alpha_start))
    # print("Alpha transition {}".format(alpha_transition))

    top_best = 3

    lm_models = arpa.loadf("ukn.3.lm")
    lm = lm_models[0]

    emission_scores = get_emission_scores(sentence_indexes, models, test_set)

    if (alpha_start and alpha_transition):
        guess = get_viterbi_sentence(emission_scores, alpha_start, alpha_transition)
    else:
        guess = list(emission_scores.idxmax(axis = 0))

    guesses.extend(guess)

    word_probabilities = [v for k, v in emission_scores.to_dict().items()]
    probabilities.extend(word_probabilities)


    logger.debug("Guess {}".format(guess))
    logger.debug("Probability {}".format(word_probabilities))


    return emission_scores


def get_emission_scores(sentence_indexes, models, test_set):
    logger = logging.getLogger('recognizer')

    aggregate = False

    min_score = 1e6 * (-1)

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
                emission_score = min_score
                # emission_score = float("-inf")

            logger.debug("Emission score {}".format(emission_score))

            if aggregate:
                # Aggregate score
                try:
                    score = scores.get_value(clean_word, score_column)
                except:
                    score = 0
                score += emission_score
                logger.debug("Score {}".format(score))
                scores.set_value(clean_word, score_column, score)

                logger.debug("Score {}".format(score))
            else:
                scores.set_value(word, score_column, emission_score)

        scores.fillna(0, inplace=True)
    return scores

def get_viterbi_sentence(scores, alpha_start = 1, alpha_transition = 1):
    logger = logging.getLogger('recognizer')

    top = 5
    min_score = 1e6 * (-1)

    lm_models = arpa.loadf("ukn.3.lm")
    lm = lm_models[0]

    states_num, observations_num = scores.shape
    states = list(scores.index)
    observations = list(scores.columns.values)

    viterbi = pd.DataFrame(index = states, columns = observations)
    backpointers = pd.DataFrame(index = states, columns = observations)

    step_0 = 0
    # Initialization step 0
    for state in states:
        emission_score = scores.get_value(state, observations[step_0])
        sentence = ['<s>']
        sentence.append(state)
        transition_score = get_n_gram_score(sentence, n_gram_model = lm, n_gram = 3)
        viterbi.set_value(state, observations[step_0], emission_score + alpha_start * transition_score)
        backpointers.set_value(state, observations[step_0], 0)

    # Recursion
    for observation in range(1, len(observations)):
        logger.debug("Observation {}".format(observation))
        # Get the last top states from previous step
        top_states = list(scores.sort_values(by = observations[observation - 1], ascending = False)[0:top].index)
        for state in states:
            # Get emission score from currente step
            emission_score = scores.get_value(state, observations[observation])
            # Get the max score emission_score + emission_score + transition_score
            for top_state in top_states:
                best_score = min_score
                best_state = None
                sentence = []
                sentence.append(top_state)
                sentence.append(state)
                # Get the transition score
                transition_score = get_n_gram_score(sentence, n_gram_model = lm, n_gram = 3)
                # Get the previous emission score
                emission_score_previous = scores.get_value(top_state, observations[observation - 1])
                # middle_score = alpha_transition * transition_score + emission_score + emission_score_previous
                middle_score = alpha_transition * transition_score + emission_score_previous
                # print("Middle score {}".format(middle_score))
                # Update the max score
                if middle_score > best_score:
                    best_score = middle_score
                    best_state = top_state
                    # print("Best score {}". format(best_score))
            state_score = best_score + emission_score
            viterbi.set_value(state, observations[observation], state_score)
            backpointers.set_value(state, observations[observation], best_state)

    # Termination
    # steps = len(observations)
    # last_state = list(viterbi.sort_values(by = observations[steps - 1], ascending = False)[0:1].index)
    # viterbi_sentence = []
    # viterbi_sentence.extend(last_state)

    # for observation in range(steps - 1, 0, -1):
    #     viterbi_sentence.append(backpointers.get_value(viterbi_sentence[steps - 1 - observation], observations[observation]))
    # return list(reversed(viterbi_sentence))
    #return viterbi

    viterbi_sentence = list(viterbi.idxmax(axis = 0))
    return viterbi_sentence


def recognize(models: dict, test_set: SinglesData, alpha_start = 0, alpha_transition = 0):
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
    # print("Alpha start {}".format(alpha_start))
    # print("Alpha transition {}".format(alpha_transition))

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
        sentence_score = get_sentence_score(sentence_indexes, models, test_set, probabilities, guesses, alpha_start, alpha_transition)
        logger.debug("Sentence indexes {}".format(sentence_indexes))
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

