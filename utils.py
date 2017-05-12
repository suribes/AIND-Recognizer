import logging

def config_log():
    # create logger with 'recognizer'
    logger = logging.getLogger('recognizer')
    logger.setLevel(logging.ERROR)
    # create file handler
    fh = logging.FileHandler('recognizer.log', mode = 'w')
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    # formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(message)s')
    # fh.setFormatter(formatter)
    # ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
