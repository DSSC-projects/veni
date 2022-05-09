import abc


class Module(abc.ABC):
    def __init__(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def forward(self, params, x):
        pass

    def __call__(self, params, x):
        return self.forward(params, x)


class Activation(abc.ABC):
    def __init__(self, f):
        self._f = f

    @staticmethod
    @abc.abstractmethod
    def forward(self, x):
        pass

    def __call__(self, x):
        return self.forward(x)
