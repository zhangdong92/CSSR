# -*- encoding: utf-8 -*-

import logging
import logging.handlers
import os
import sys

LOG_BASIC_FORMAT = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)s] %(message)s"
LOG_PATH="log/CSSR.log"

def init_logger(log_file=None):
    if len(logging.root.handlers) == 0:
        formatter = logging.Formatter(LOG_BASIC_FORMAT)
        log_file=LOG_PATH
        if log_file:
            handler = logging.handlers.RotatingFileHandler(log_file,
                                                           maxBytes=10 * 1024 * 1024,
                                                           backupCount=100)
            handler.setFormatter(formatter)
            logging.root.addHandler(handler)

        console=logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logging.root.addHandler(console)


        logging.BASIC_FORMAT = LOG_BASIC_FORMAT
        logging.root.setLevel(logging.INFO)
        # logging.info("Init logger success.")
    return logging.root
def get_log():
    print(os.getcwd())
    path=os.path.abspath(sys.path[0])+"/" + LOG_PATH
    index=path.rfind("\\")
    index2=path.rfind("/")
    # print "index="+str(index)+" index2="+str(index2)
    index=index if index>index2 else index2
    parent_path=path[:index]
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    return init_logger(path)

log=get_log()