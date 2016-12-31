from scipy import spatial
import pandas as pd

'''
    KNN - K Nearest Neighbours
        Lets say you have a set of data points (training data) , and you want to predict for data point 'x',
        you figure out the right value of 'k' and find its nearest neighbours (distance metric varies based on problem)
'''


class KNN:

    def __init__(self):
        df = pd.read_csv('collab-filtering-data/u.data', names=['user_id', 'movie_id', 'rating'], usecols=range(3))
        print df.head(n=5)

KNN()