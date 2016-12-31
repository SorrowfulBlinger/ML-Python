import pandas as pd
import numpy as nm

'''
    Recommender Sytems
        User Based Collab Filtering
            - Identify similar users and recommned things used by similar people but u haent used
        Item Based
            - Recommend similar items
            - Faster as there are less utems than users to compare
            - Difficult to manipulate
            - NOt affected by Users fickle behaviour / changing behaviours like for some time they watch SciFI but then comendy so we should not refer comedy to similar users
'''


class ItemBasedCollabFiltering:
    def __init__(self):
        coll_names = ['user_id', 'movie_id', 'rating']
        df1 = pd.read_csv('collab-filtering-data/u.data', sep='\t', names=coll_names, usecols=range(3))

        movie_names = ['movie_id', 'movie_name']
        df2 = pd.read_csv('collab-filtering-data/u.item', sep='|', names=movie_names, usecols=range(2))
        self.__movie_ratings = pd.merge(df1, df2)

    def movies_similar_to_star_wars(self):
        movie_ratings_pivot = self.__movie_ratings.pivot_table(index='user_id', columns='movie_name', values='rating')
        star_wars = movie_ratings_pivot['Star Wars (1977)']

        #  if Correlte userids/rows - user based collab filering
        similar_movies = movie_ratings_pivot.corrwith(star_wars)
        similar_movies = similar_movies.dropna()
        self.__base_similar_movies = similar_movies.sort_values(ascending=False)
        self.__base_similar_movies = pd.DataFrame(self.__base_similar_movies, columns=['similarity'])
        self.__base_similar_movies['movie_name'] = self.__base_similar_movies.index
        # print self.__base_similar_movies.columns.values
        df = self.__movie_ratings.groupby('movie_name', as_index=False).agg({'rating': [nm.size, nm.mean]})
        df.columns = df.columns.droplevel()
        df.columns = ['movie_name', 'rating_count', 'mean_rating']
        df = df[df['rating_count'] >= 100]
        df = df.join(self.__base_similar_movies.set_index('movie_name'), how='inner', on='movie_name')
        df = df.sort_values(['similarity'], ascending=False)
        # Movies similar to star wars
        print df[['movie_name', 'similarity', 'rating_count', 'mean_rating']].head(n=3)

    def recommend_movies_for(self, user_id):
        movie_ratings_pivot = self.__movie_ratings.pivot_table(index='user_id', columns='movie_name', values='rating')
        corr_result = movie_ratings_pivot.corr(method='pearson', min_periods=100)
        corr_result.dropna()
        user_ratings = movie_ratings_pivot.loc[user_id].dropna()

        top_recommendations = pd.Series()
        user_watched_movies = []
        for movie, rating in user_ratings.iteritems():
            similar_movies = corr_result[movie].dropna()
            score_in_relation_to_curr_movie = similar_movies.map(lambda (x): x * rating)
            top_recommendations = top_recommendations.append(score_in_relation_to_curr_movie)
            user_watched_movies.append(movie)

        # Group by movie Name as it would be related multiple moovies user has seen
        top_recommendations = top_recommendations.to_frame().groupby(top_recommendations.index).sum()
        top_recommendations = top_recommendations.loc[~top_recommendations.index.isin(user_watched_movies)]
        top_recommendations.columns = ['score']

        # Select the movies with max score
        top_recommendations = top_recommendations.sort_values('score', ascending=False)
        print top_recommendations.head(n=10)
        print top_recommendations.loc[user_watched_movies]


pd.options.display.max_colwidth = 40
icb = ItemBasedCollabFiltering()
# icb.movies_similar_to_star_wars()
icb.recommend_movies_for(0)
