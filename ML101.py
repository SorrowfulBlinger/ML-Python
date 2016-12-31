from __future__ import division
import numpy as nm
from scipy import stats
from array import array
from scipy.stats import norm, expon, binom, poisson
import matplotlib.pyplot as plot
import random
import collections


'''
Types of data to analyse
    Numerical
        Discrete - ages of population
        Continuos -  time spent on pages before exit
    Categorical - Geography classification based on population
    Ordinal - Mixture of above 2. ex ratings (1-5 stars) - They denote

'''

'''
    Variance of population = Mean(Square of Diff between Sample & Mean)
    Variance of sample = Sum(Square of Diff between Sample & Mean)/N-1

    Square is done  in variance
        to take into consideration negative deviations or else it would just cancel out positive ones
        empahsise outliers
    data = [1, 4, 4]
    StdDev = sqrt(variance)
    Based on how far the data points are wrt to StdDev it could be classified as outlier
'''

'''
    Prob density fucntion(pdf) - continuos data  specifies prob of event occuring in any range
    Prob Mass Func(pmf) - Same ass abovee but discrete data
    Poisson dist is

'''

'''
    Percentile - %of points below xth percentile in a distribution
    Moments - Shape of distribution
        1st moment = mean
        2nd = variance
        3rd = skew (how lopsided tail is )
        4th - kurtosis (how sharp peak is)

'''

'''
    Covariance - elation between 2 vectors
    Correlation [-1, 1] - same as above but normalised
'''

'''
    Bayes Theorem P(A|B) = (P(B|A) * P(A))/P(B) -> P(A|B) different from P(B|A)
    Conditional probability P(A|B) = P(A,B)/P(B)
'''

class StatsUtils:

    def __init__(self):
        __temp = ''

    def hai(self):
        return 'instance method'

    @staticmethod
    def covariance(x, y):
        mean_x = nm.mean(x)
        mean_y = nm.mean(y)

        vect_x = [xi - mean_x for xi in x]
        vect_y = [yi - mean_y for yi in y]
        return nm.dot(vect_x, vect_y) / (len(x)-1)

    @staticmethod
    def correlation(x, y):
        return StatsUtils.covariance(x, y) / (nm.std(x) / nm.std(y))

class ConditionalProbability:

    # class variables
    __classVariable = 1

    # Instance variables
    def __init__(self):
        self.__purchase = {10: 0, 20: 0, 30: 0, 40: 0}
        self.__age = {10: 0, 20: 0, 30: 0, 40: 0}
        self.__total_purchases = 0
        self.__total_space = 0

    def generate_data(self, dependent, total):
        self.__total_space = total
        for _ in range(total):
            rand_age = random.choice([10, 20, 30, 40])
            independent_purchase_probability = {age: random.random() for age in sorted(self.__age.keys())}
            dependent_purchase_probability = {age: age/100 for age in sorted(self.__age.keys())}
            self.__age[rand_age] += 1;
            if not dependent:
                if random.random() < independent_purchase_probability[rand_age]:
                    self.__purchase[rand_age] += 1
                    self.__total_purchases += 1
            else:
                if random.random() < dependent_purchase_probability[rand_age]:
                    self.__purchase[rand_age] += 1
                    self.__total_purchases += 1

        print 'Total Space-', self.__total_space
        print 'Total purchase-', self.__total_purchases
        print 'Age Distribution-', self.__age
        print 'Age-purchase Distribution-', self.__purchase

    def get_conditional_prob(self, age):
        # P(Purchase by age | Prob of age) =  P(Purchase , age) / P(age)
        return (self.__purchase[age]/self.__total_space) / ((self.__age[age]) / self.__total_space)

    def get_purchase_prob(self):
        return self.__total_purchases/ self.__total_space

    def get_plot(self):
        self.__age = collections.OrderedDict(sorted(self.__age.items()))
        self.__purchase = collections.OrderedDict(sorted(self.__purchase.items()))
        print self.__age, self.__purchase

        plot.xlabel('Age')
        plot.ylabel('Purchase Count')
        plot.plot(range(len(self.__age.keys())), self.__purchase.values())
        plot.xticks(range(len(self.__age.keys())), list(self.__age.keys()))
        plot.show()
        plot.close()


def ml101():

    print StatsUtils().hai()
    dist = nm.random.normal(2000, 100, 100)
    print 'Mean-', nm.mean(dist)
    print 'Median-', nm.median(dist)
    print 'Mode-', stats.mode(dist)
    print 'Mode-', stats.mode(nm.random.randint(1800, 2000, 20))

    x_axis = nm.arange(0, 10, 0.1)
    plot.plot(x_axis, norm.pdf(x_axis))
    #plot.show()
    plot.close()

    plot.plot(x_axis, binom.pmf(x_axis, 10, 0.5))
    #plot.show()
    plot.close()

    plot.plot(x_axis, expon.pdf(x_axis))
    #plot.show()
    plot.close()

    x_axis = nm.arange(400, 600, 1)
    # Given range of values 0-10 , with mean 5, pobablity of getting any other values can be calculate from this curve
    plot.plot(x_axis, poisson.pmf(x_axis, 500))
    #plot.show()
    plot.close()

    # Should be close to 100 as specified while creating normal dist
    print 'StdDev-', nm.std(dist)
    print 'Variance-', nm.var(dist)
    plot.hist(dist, 10)
    #plot.show()
    plot.close()

    #All values less than below constitue 99%
    print '99th percentile ', nm.percentile(dist, 99)

    #Moments
    print '1st Moment-', nm.mean(dist)
    print '2st Moment-', nm.var(dist)
    print '3st Moment-', stats.skew(dist)
    print '4st Moment-', stats.kurtosis(dist)

    print 'Covariance-', StatsUtils.covariance([1, 2, 0],[2, 3, 4])
    print 'Numpy Covariance', nm.cov([1, 2, 0], [2, 3, 4])

    print 'correlation-', StatsUtils.correlation([1, 2, 0], [2, 3, 4])
    print 'Numpy correlation', nm.corrcoef([1, 2, 0], [2, 3, 4])

    # Add outlier
    dist = nm.append(dist, [30000000])
    print 'Mean-', nm.mean(dist)

    # Median is a better estimator when huge outliers exist in sample data
    print 'Median-', nm.median(dist)
    print 'Mode-', stats.mode(nm.random.randint(1800, 2000, 20))
    print 'StdDev-', nm.std(dist)
    print 'Variance-', nm.var(dist)

    # Homogenous container
    print array('i', [2, 3, 11])

    # Understand relation between customer purchase and age
    # Independent events P(A|B) = P(A)
    print '*** Understand relation between customer purchase and age ***'
    cp = ConditionalProbability()
    cp.generate_data(False, 10000)
    print 'Prob of purchase given 30 age - ', cp.get_conditional_prob(30)
    print 'Prob of purchase - ', cp.get_purchase_prob()
    cp.get_plot()

    print '*** Understand relation between customer purchase and age with dependent data ***'
    # Understand relation between customer purchase and age Dependent data
    cp = ConditionalProbability()
    cp.generate_data(True, 10000)
    print 'Prob of purchase given 30 age - ', cp.get_conditional_prob(30)
    cp.get_plot()

ml101()

