import logging
import datetime
import time
import os
import sys

def get_handlers(log_file=True, log_stdout=True):
    handlers = []
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%m-%d-%y')
    if not os.path.isdir(os.path.join('.', 'log')):
        os.mkdir(os.path.join('.', 'log'))
    log_filepath = os.path.join('.', 'log', date + '.log')

    log_fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    if log_file:
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(log_fmt)
        handlers.append(file_handler)
    if log_stdout:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_fmt)
        handlers.append(console_handler)

    return handlers

def set_logger(name='root', log_file=True, log_stdout=True):
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%m-%d-%y')
    if not os.path.isdir(os.path.join('.', 'log')):
        os.mkdir(os.path.join('.', 'log'))
    log_filepath = os.path.join('.', 'log', date + '.log')

    log_fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    root_logger = logging.getLogger(name)
    if log_file:
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(log_fmt)
        root_logger.addHandler(file_handler)
    if log_stdout:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_fmt)
        root_logger.addHandler(console_handler)


def get_logger(name='root', log_file=True, log_stdout=True):
    set_logger(name, log_file, log_stdout)
    return logging.getLogger(name)
