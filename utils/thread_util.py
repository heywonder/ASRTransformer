# coding=utf8
import threading


class safethread_iter:
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def safethread_generator(f):
    def g(*a, **kw):
        return safethread_iter(f(*a, **kw))
    return g
