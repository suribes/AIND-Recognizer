import warnings
from asl_data import SinglesData
import logging


def recognize(models: dict, test_set: SinglesData):
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

    probabilities = []
    guesses = []
    # TODO implement the recognizer

    logger.info("Models {}".format(models))
    # logger.info("test_set data {}".format(test_set._data))

    sequences = test_set.get_all_sequences()
    # Xlengths = test_set.get_all_Xlengths()

    logger.info("Sequences {}".format(sequences))
    # logger.info("Xlengths {}".format(Xlengths))
    # logger.info("Test set length {}".format(len(test_set)))
    logger.info("Test words {}".format(test_set.wordlist))
    for test_word_index, test_word in enumerate(test_set.wordlist):
        logger.info("test_word {}".format(test_set.wordlist[test_word_index]))
        logger.info("test_word {}".format(test_word))
        logger.info("test_word_index {}".format(test_word_index))
        logger.info("get_item_Xlengths {}".format(test_set.get_item_Xlengths(test_word_index)))
        test_X, test_lenghts = test_set.get_item_Xlengths(test_word_index)
        logger.info("test_X {}".format(test_X))
        logger.info("test_lenghts {}".format(test_lenghts))

        scores = []
        probability = {}
        best_score = float("-inf")
        guess = None
        for word, model in models.items():
            logger.info("Model {}".format(model))

            try:
                score = model.score(test_X, test_lenghts)
                scores.append(score)
                logger.info("Score: {}".format(score))
            except:
                score = float("-inf")
                scores.append(score)
            probability.update({word: score})
            if score > best_score:
                best_score = score
                guess = word
        logger.info("Scores {}".format(scores))
        # probabilities.append(scores)

        # max_value = max(scores)
        # max_index = scores.index(max_value)
        guesses.append(guess)
        probabilities.append(probability)


    logger.info("Probabilities {}".format(probabilities))
    logger.info("Guesses {}".format(guesses))
    return probabilities, guesses
    # raise NotImplementedError

if __name__ == "__main__":
    from  asl_test_recognizer import TestRecognize
    from utils import config_log

    config_log()
    logger = logging.getLogger('recognizer')
    test_model = TestRecognize()
    test_model.setUp()
    test_model.test_recognize_probabilities_interface()
    # test_model.test_recognize_guesses_interface()
