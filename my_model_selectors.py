import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from asl_utils import combine_sequences
import logging



class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        logger = logging.getLogger('recognizer')
        train_X = self.X
        train_lengths = self.lengths
        best_model_BIC_score = float('inf')
        best_model = None

        logger.debug("Sequence {}".format(self.X))
        logger.debug("Features {}".format(train_X[0]))
        logger.debug("Number of features {}".format(len(train_X[0])))

        # Get number of features
        d = len(train_X[0])

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            logger.debug("n_components: {}".format(n_components))

            try:
                # Get nubmer of states
                n = n_components

                model = GaussianHMM(n_components=n_components, n_iter=1000).fit(train_X, train_lengths)

                # Calculate score
                score = model.score(train_X, train_lengths)

                # Calculate the number of parameters p
                p = n*n + 2*d*n - 1

                # Calculate BIC score
                BIC_score = (-2 * score) + (p * math.log(len(train_lengths)))
                logger.debug("Score: {}".format(score))
                logger.debug("BIC_Score: {}".format(BIC_score))
            except:
                continue

            # Check wheter we have a better model
            if BIC_score < best_model_BIC_score:
                best_model_BIC_score = BIC_score
                best_model = model
                logger.debug("New best Model: {}".format(best_model))
            logger.debug("Avg Score: {}".format(BIC_score))

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        logger = logging.getLogger('recognizer')

        train_X = self.X
        train_lengths = self.lengths
        best_model_DIC_score = float('-inf')
        best_model = None
        logger.info(self.this_word)

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            logger.debug("n_components: {}".format(n_components))

            try:
                model = GaussianHMM(n_components=n_components, n_iter=1000).fit(train_X, train_lengths)
                score = model.score(train_X, train_lengths)
                # DIC_score = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

                other_words_score = score * -1
                for word in self.words:
                    otherX, otherlenghts = self.hwords[word]
                    other_words_score += model.score(otherX, otherlenghts)
                avg_others_score = other_words_score / (len(self.hwords) - 1)
                DIC_score = score - avg_others_score

                logger.debug("Score: {}".format(score))
                logger.debug("Avg others score: {}".format(avg_others_score))
                logger.debug("DIC_Score: {}".format(DIC_score))
            except:
                continue

            if DIC_score > best_model_DIC_score:
                best_model_DIC_score = DIC_score
                best_model = model
                logger.debug("New best Model: {}".format(best_model))
            logger.debug("Avg Score: {}".format(DIC_score))

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        logger = logging.getLogger('recognizer')
        logger.info("--- STARTED SELECTOR CV ---")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection using CV
        best_model_avg_score = float('-inf')
        best_model = GaussianHMM(n_components=self.n_constant, n_iter=1000).fit(self.X, self.lengths)

        word = self.this_word
        logger.info(word)
        word_sequences = self.sequences

        split_method = KFold()

        for n_components in range(self.min_n_components, self.max_n_components + 1):
            scores = []
            logger.debug("n_components: {}".format(n_components))

            try:
                for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
                    train_X, train_lengths = combine_sequences(cv_train_idx, word_sequences)
                    test_X, test_lengths = combine_sequences(cv_test_idx, word_sequences)

                    # Generate the model and calculate the score
                    model = GaussianHMM(n_components=n_components, n_iter=1000).fit(train_X, train_lengths)
                    score = model.score(test_X, test_lengths)
                    scores.append(score)
                    logger.debug("Score: {}".format(score))
            except:
                continue

            # Check wheter we have a better model
            if len(scores) > 0:
                avg_score = sum(scores) / len(scores)
                if avg_score > best_model_avg_score:
                    best_model_avg_score = avg_score
                    best_model = model
                    logger.debug("New best Model: {}".format(best_model))
                logger.debug("Avg Score: {}".format(avg_score))

        return best_model


if __name__ == "__main__":
    from  asl_test_model_selectors import TestSelectors
    from utils import config_log

    config_log()
    logger = logging.getLogger('recognizer')
    test_model = TestSelectors()
    test_model.setUp()
    # test_model.test_select_cv_interface()
    # test_model.test_select_bic_interface()
    test_model.test_select_dic_interface()
