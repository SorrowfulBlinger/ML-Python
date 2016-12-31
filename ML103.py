import os
import io
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as nm
import random
from sklearn.preprocessing.data import scale
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier
import pydotplus

'''
    Supervised Learning - Results for data known , use that to predict values for new samples
    Unsupervised Learning - REsults are not known, use data to figure out differnet relations /classes etc - Used tp figure out unknown relatioship in data
    K Fold strategy - Divide sample data into 'k' segments use 'k-1' to train model , and '1' as test data
'''

'''
    Spam classifier using Naive Bayes
    P(Spam Message | Message has Word 'X') =  (P(Word 'X' in message | Spam Message) * P(Spam Message))/ P(Word 'X' in message)
    P(A|B) = P(A,B)/P(B)
    P(B|A) = P(A,B)/P(A)
    SO P(A|B) = (P(B|A)*P(A))/P(B)
    P(B) = P(B|A)*P(A) + P(B|!A)*P(!A)
    P(Spam Message) =  for all words X1 ... Xn in a message P(Spam Message | Word 'X1') *  P(Spam Message | Word 'X2') * ...
    Called Naive cos words are independent (single words have contribution , not taking combination of words)
'''

'''
    KMeans clustering unsupervised - clusters data into k clusters
    Challenges
        - You need to figure out what k should be - square of diff of errors should be min
        - Local Minima - so run with different initial centroid to see if clusters match
        - Type of classification , based on what is not determined by kmeans , yourself have to figure out relation in data
        - Normalise data points
'''

'''
    Deciscion Trees(DT) - Flowchart with decisions on attribute at different levels - at each level we choose an attribue that lowers entropy for next step( ir attribute able to classify most of data)
    Supervidesd learning
    ex - Whether to hire a person based on attributes (Education, currentEmployer, yearsOfExp) etc
        if(currentEmplyer == MNC )
            Hire
        else
            if(yearsOfExp >5)
                Hire
            else
                if(Eductaion == IIT)
                    Hire
                else
                    Reject

    Problems - Overfitting
    Random Forest - Multiple DT - Bootstrap Aggregating/'Baging' samples of datasets , make decision based on votes of all trees
'''

'''
    Ensemble learning  multiple models & vote the best among them like RandomForst, kfold k means
    Bucket of models / stacking / boosting refers to same
    Always advisible to use ensemble over complicated models(overfit)
'''

'''
    SVM - used when multiple features to be made use , computationlay expensive, Supervised learning
    different kernels
'''


class SVM:
    def __init__(self, n):
        self.__generate_data(n)
        self.__predict(int(0.8 * n), n)

    @staticmethod
    def __generate_kstar_data(rating):
        return {

            1: {'post_count': random.randint(500, 600), 'present_since': random.randint(1, 10),
                'absent_days': random.randint(0, 1), 'videos_count': random.randint(500, 900),
                'rating': 5}
            ,
            2: {'post_count': random.randint(300, 400), 'present_since': random.randint(20, 40),
                'absent_days': random.randint(5, 10), 'videos_count': random.randint(300, 350),
                'rating': 4}
            ,
            3: {'post_count': random.randint(100, 200), 'present_since': random.randint(50, 100),
                'absent_days': random.randint(50, 100), 'videos_count': random.randint(100, 200),
                'rating': 3}
            ,
            4: {'post_count': random.randint(50, 100), 'present_since': random.randint(100, 1000),
                'absent_days': random.randint(200, 500), 'videos_count': random.randint(50, 90),
                'rating': 2}
            ,
            5: {'post_count': random.randint(0, 30), 'present_since': random.randint(1000, 100000),
                'absent_days': random.randint(500, 5000), 'videos_count': random.randint(0, 10),
                'rating': 1}
            ,
            0: {'post_count': random.randint(0, 10000), 'present_since': random.randint(0, 100000),
                'absent_days': random.randint(0, 10000), 'videos_count': random.randint(0, 10000),
                'rating': random.choice(['1', '2', '3', '4', '5'])}

        }[rating]

    def __predict(self, test_index_start, test_index_end):
        total = 0
        success = 0
        for index in range(test_index_start, test_index_end):
            if self.__clf.predict(nm.array(self.__data_frame.loc[index, self.__features]).reshape(1, -1)) == \
                    self.__data_frame.loc[index, 'rating']:
                success += 1
            total += 1
        print 'Success Ratio = ', success * 100 / total, '%'

    def __generate_data(self, n):
        # Rate a user from 1-5 based on #active_days #posts, #absent_days, #videos , #friends
        data = []
        index = []
        for i in range(n):
            if i % int((n/30)+19) == 0:
                rating_to_get = 0
            else:
                rating_to_get = random.randint(1, 5)
            data.append(SVM.__generate_kstar_data(rating_to_get))
            index.append(i)
        self.__data_frame = pd.DataFrame(data, index=index)
        print self.__data_frame.head(n=5)

        self.__features = ['post_count', 'present_since', 'absent_days', 'videos_count']
        # We have some good data , lets use SVM to classify
        # diff types of kernel 
        self.__clf = svm.SVC(kernel='poly', C=1.0).fit(self.__data_frame.loc[:.8 * n, self.__features],
                                                         self.__data_frame.loc[:.8 * n, 'rating'])


class RandomForest:
    def __init__(self, dt):
        # Use 10 DTs
        clf = RandomForestClassifier(n_estimators=10)
        clf = clf.fit(dt.data_frame[dt.features], dt.data_frame['Hired'])
        print clf.predict([[10, 1, 2, 1, 1, 1], [20, 1, 2, 2, 0, 1]])


class DecisionTree:
    def __init__(self):
        self.__data_frame = pd.read_csv('PastHires.csv', header=0)
        mapper = {'Y': 1, 'N': 0}
        mapper_edu = {'BS': 0, 'MS': 1, 'PhD': 2}
        self.__features = list(self.__data_frame.columns[:6])
        self.__data_frame['Employed?'] = self.__data_frame['Employed?'].map(mapper)
        self.__data_frame['Level of Education'] = self.__data_frame['Level of Education'].map(mapper_edu)
        self.__data_frame['Top-tier school'] = self.__data_frame['Top-tier school'].map(mapper)
        self.__data_frame['Interned'] = self.__data_frame['Interned'].map(mapper)
        self.__data_frame['Hired'] = self.__data_frame['Hired'].map(mapper)

        self.__clf = tree.DecisionTreeClassifier()
        self.__clf.fit(self.__data_frame[self.__features], self.__data_frame['Hired'])
        self.__print_tree()

    def __print_tree(self):
        dot_data = tree.export_graphviz(self.__clf, out_file=None, feature_names=self.__features,
                                        filled=True, rounded=True,
                                        special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf("dt-output.pdf")

    @property
    def features(self):
        return self.__features

    @property
    def data_frame(self):
        return self.__data_frame

    # Dont use this just to show setter
    @features.setter
    def features(self, value):
        pass


class KMeansClustering:
    def __init__(self, n, k):
        self.__data_points = None
        # Generate data
        self.__create_fake_clustered_data(n, k)

        # Cluster data
        self.__cluster(k)

    def __create_fake_clustered_data(self, n, k):
        data = []
        n_per_cluster = float(n) / k
        # Lets say we cluster income vs age
        for _ in range(k):
            income_centroid = random.randint(100000, 2000000)
            age_centroid = random.randint(20, 60)
            for _ in range(int(n_per_cluster)):
                data.append([nm.random.normal(income_centroid, 50000), nm.random.normal(age_centroid, 3)])
        self.__data_points = nm.array(data)

    def __cluster(self, k):
        model = KMeans(n_clusters=k)
        model.fit(scale(self.__data_points))
        print model.labels_

        plt.scatter(self.__data_points[:, 0], self.__data_points[:, 1], c=model.labels_.astype(nm.float))
        plt.show()
        plt.close()


# spam / ham (non spam emails)
class NaiveBayesSpamClassifier:
    def __init__(self, emails_path):
        self.__classifier = None
        self.__vectorizer = None
        # Read all emails and put inside pandas data frame
        rows = []
        indexes = []
        for path in emails_path:
            for root, dirnames, filenames in os.walk(path):
                classifier = 'ham'
                if 'spam' in path:
                    classifier = 'spam'

                for file in filenames:
                    full_path = os.path.join(root, file)
                    f_handle = io.open(full_path, 'r', encoding='latin1')
                    email_content_starts = False
                    contents = []
                    for line in f_handle:
                        if line == '\n':
                            email_content_starts = True
                        if email_content_starts:
                            contents.append(line)
                    f_handle.close()
                    rows.append({'Message': '\n'.join(contents), 'Class': classifier})
                    indexes.append(full_path)

        # Add rows into data frame
        self.__data_frame = pd.DataFrame(rows, index=indexes)
        # print self.__data_frame.head(n=1)
        self.naive_bayesian_run()

    def predict(self, emails):
        print dict(zip(emails, self.__classifier.predict(self.__vectorizer.transform(emails))))

    def naive_bayesian_run(self):
        self.__vectorizer = CountVectorizer()
        counts = self.__vectorizer.fit_transform(self.__data_frame['Message'].values)
        classes = self.__data_frame['Class'].values
        self.__classifier = MultinomialNB()
        self.__classifier.fit(counts, classes)


# NaiveBayesSpamClassifier(['emails/spam', 'emails/ham']).predict(['Viagra !!', 'earn dollars', 'hello world tommorrow'])
# KMeansClustering(100, 4)
# dt = DecisionTree()
# RandomForest(dt)
SVM(1000)
