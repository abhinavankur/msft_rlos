from numpy.random import seed
from numpy.random import randn
from scipy.stats import shapiro


class NormalTestError(Exception):
    pass


class NewTrainer:
    ...
    def train(self, x: list[list[float]], y: list[float]):
        seed(1)
        alpha = 0.05
        stat, p = shapiro(x)
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0)')
            raise NormalTestError('Sample looks Gaussian (fail to reject H0)')
        else:
            print('Sample does not look Gaussian (reject H0)')
        ...

    def predict(self, x: list[float]) -> float:
        print(type(x))
        if not isinstance(x, list):
            raise TypeError('List of floats needed for prediction')
        for i in range(len(x)):
            if not isinstance(x[i], float):
                raise TypeError('Item {} is not a float'.format(i))
        return x

if __name__ == "__main__":
    obj = NewTrainer()
    obj.predict(9)
