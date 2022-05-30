import pandas as pd
from pandas import DataFrame
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.svm import SVC
from typing import Tuple, Any, Union
from typing import List
from sklearn.model_selection import cross_val_score


def get_data_from_file() -> Tuple[List[str], np.array]:
    data: Union[dict[Any, DataFrame], DataFrame] = pd.read_excel(r'C:\Users\barte\Desktop\semestr6\SIe4\DaneMartynki.xlsx')
    target: np.array = data.pop('Gatunek').values
    data: List[str] = list(data.pop('Tekst').values)
    return data, target


def naive_bayes(data: List[str], target: np.array, alpha: float, tf: bool, idf: bool):
    transformer = TfidfTransformer(use_idf=idf) if tf else None
    model = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', transformer),
        ('clf', MultinomialNB(alpha=alpha, fit_prior=False)),
    ])
    score = cross_val_score(model, data, target, cv=10)
    return score


def svm(data: List[str], target: np.array, alpha: float, tf: bool, idf: bool):
    #tf - zmiana z liczby na czestotliwosc, idf - wykasowanie slow pojawiajacych sie wszedzie
    transformer = TfidfTransformer(use_idf=idf) if tf else None
    model = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', transformer),
        ('clf', SVC(C=1, gamma='scale')),
    ])
    score = cross_val_score(model, data, target, cv=10)
    return score


if __name__ == '__main__':
    data, target = get_data_from_file()
    print(target)
    # result = naive_bayes(data, target, 0.2, False, False)
    # print('naive bayes:', result.mean())
    result = svm(data, target, 1e-3, True, True)
    print('svm:', result.mean())



