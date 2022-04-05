import pytest
from training import NewTrainer, NormalTestError

from numpy.random import seed
from numpy.random import randn, uniform
from scipy.stats import shapiro



obj = NewTrainer()
seed(1)
def test_list():
    const = 9
    # obj = NewTrainer()
    with pytest.raises(TypeError):
        obj.predict(const)

def test_item_valid():
    # obj = NewTrainer()
    x = [1.0, 1.1, 1.2, 1.3]
    assert obj.predict(x) == x

def test_item_invalid():
    # obj = NewTrainer()
    x = [1, 2.3, 'str', 4.4]
    with pytest.raises(TypeError):
        obj.predict(x)

def test_normality():
    # obj = NewTrainer()
    alpha = 0.05
    data = 5 * randn(100) + 50
    y = randn(100)
    with pytest.raises(NormalTestError):
        obj.train(data, y)

def test_normality_invalid():
    data = uniform(-5, 5, 100)
    y = randn(100)
    with pytest.raises(NormalTestError):
        obj.train(data, y)


def test_columns():
    pass