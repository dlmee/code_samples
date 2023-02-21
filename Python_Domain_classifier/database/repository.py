import abc
from model.inflections import *
from model.vectors import *


class Repository(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def abstract(self):
        raise NotImplementedError

