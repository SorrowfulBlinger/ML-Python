from __future__ import print_function
from scipy import spatial
import pandas as pd
import numpy as nm

'''
    KNN - K Nearest Neighbours
        Lets say you have a set of data points (training data) , and you want to predict for data point 'x',
        you figure out the right value of 'k' and find its nearest neighbours (distance metric varies based on problem)


                          names=['movie_id', 'movie_name', 'genre1', 'genre2', 'genre3', 'genre4', 'genre5', 'genre6',
                                 'genre7', 'genre8', 'genre9', 'genre10', 'genre11', 'genre12'],

                          #usecols=[1, 2] + range(5, 22))
'''


class KNN:
    def __init__(self):
        df = pd.read_csv('collab-filtering-data/u.data', sep='\t', names=['user_id', 'movie_id', 'rating'],
                         usecols=range(3))
        df = df.groupby('movie_id').agg({'rating': [nm.size, nm.mean]})
        df.columns = df.columns.droplevel(0)
        df.columns = ['rating_count', 'rating_avg']

        # names=['movie_id','movie_name','genre']
        df1 = pd.read_csv('collab-filtering-data/u.item', sep='|',
                          names=['movie_id', 'movie_name', 'genre1', 'genre2', 'genre3', 'genre4', 'genre5', 'genre6',
                                 'genre7', 'genre8', 'genre9', 'genre10', 'genre11', 'genre12', 'genre13', 'genre14',
                                 'genre15', 'genre16', 'genre17', 'genre18', 'genre19'],
                          usecols=[0, 1] + range(5, 24))

        df = pd.merge(df, df1, left_index=True, right_on='movie_id')
        self.__movies_info = df
        #print df.head(n=5)
        self.k_nearest_neighbours(df.loc[0], 10)

    def __compute_distance(self, movie1, movie2):
        min_rating_count = self.__movies_info['rating_count'].min()
        max_rating_count = self.__movies_info['rating_count'].max()
        range_rating = max_rating_count - min_rating_count

        movie1['rating_count'] = movie1['rating_count'].apply(lambda x: (x - min_rating_count) / float(range_rating))
        movie2['rating_count'] = movie2['rating_count'].apply(lambda x: (x - min_rating_count) / float(range_rating))
        #print movie1.head(n=5)


    def k_nearest_neighbours(self, movie_in_contention, k):
        min_rating_count = self.__movies_info['rating_count'].min()
        max_rating_count = self.__movies_info['rating_count'].max()
        range_rating = max_rating_count - min_rating_count

        movie1_rating = movie_in_contention['rating_avg']
        self.__movies_info = self.__movies_info.drop(self.__movies_info[self.__movies_info['movie_name'] == movie_in_contention['movie_name']].index)
        df = pd.DataFrame()
        # Normalise ratings
        self.__movies_info['k_distance'] = self.__movies_info.apply(
            lambda row: (((row['rating_count'] - min_rating_count) / float(range_rating)) + abs(
                row['rating_avg'] - movie1_rating) +
                (spatial.distance.cosine(row[5:24].values, movie_in_contention[5:24].values))
            ), axis=1)

        # self.__movies_info['k_distance'] = self.__movies_info.apply(
        #     lambda row: print (row[5:19].values)
        #                  )

        self.__movies_info = self.__movies_info.drop(self.__movies_info[self.__movies_info['rating_count'] < 100].index)
        sorted_list = self.__movies_info.sort_values(['k_distance'], ascending=True)
        print ('Movies similar to ', movie_in_contention['movie_name'], 'are\n', sorted_list[['movie_name', 'k_distance', 'rating_count', 'rating_avg']].head(n=k))
        print (movie_in_contention['movie_name'], 'actual rating', movie_in_contention['rating_avg'] , ', predicted rating ', nm.mean(sorted_list['rating_avg'].head(n=k)) )
        # return spatial.distance.cosine(movie1, movie2)


pd.options.display.max_colwidth = 30
KNN()
