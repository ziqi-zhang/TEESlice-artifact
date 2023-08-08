import os, sys
import datetime


class Logger(object):
    def __init__(self, log2file=False, mode='train', path=None):
        if log2file:
            assert path is not None
            fn = os.path.join(path, '{}-{}.log'.format(mode, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
            self.fp = open(fn, 'w')
        else:
            self.fp = sys.stdout

    def add_line(self, content):
        self.fp.write(content+'\n')
        self.fp.flush()
