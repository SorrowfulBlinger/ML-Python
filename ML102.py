from scipy import stats
from sklearn.metrics import r2_score
import numpy as nm
import matplotlib.pyplot as plot
import random
import pandas as pd
import statsmodels.api as sm


class LinearRegression:

    def __init__(self):
        pageTime = nm.random.normal(100, 10, 100)
        purchase_amount = [1000 - (3 * (x + random.randint(40, 50))) for x in pageTime]
        plot.scatter(pageTime, purchase_amount)
        self.__slope, self.__intercept, self.__r, a, b = stats.linregress(pageTime, purchase_amount)
        plot.plot(pageTime, self.get_y_value(pageTime), c='r')
        plot.show()
        plot.close()
        print 'Coeff determination should be -> 1 , means linear fit is able to fit on either sides', self.get_coeff_determination()

    def get_y_value(self, x):
        return self.__slope * x + self.__intercept

    def get_coeff_determination(self):
        return self.__r ** 2


class PolynomialRegression:

    def __init__(self):
        access_time = nm.random.normal(40, 5, 100)
        purchase_amount = nm.random.normal(1000, 50, 100)/access_time
        plot.scatter(access_time, purchase_amount)

        # Fit with Polynomial of say degree 4, make sure you dont under/overfit

        yp4 = nm.poly1d(nm.polyfit(access_time, purchase_amount, 4))
        x = [m for m in range(20, 70)]

        plot.plot(x, yp4(x), c='r')
        plot.show()
        plot.close()

        # Get R2 - Coeff of determination , should be ~1 but this is only for training data sets
        print 'Coeff determination should be -> 1 , means polynomial fit is able to fit on either sides', r2_score(purchase_amount, yp4(access_time))


'''
    Multi Variables whihc are independent
    Stderr denotes how much variation is there in y between actual n model
    R2 denotes how good the model can fit to dynamic data
'''


class MultiVariate:


    def __init__(self):
        data_frame = pd.read_excel('cars.xlsx')
        data_frame.head()

        data_frame['Make_ord'] = pd.Categorical(data_frame.Make).codes
        x1 = data_frame[['Mileage', 'Cylinder', 'Doors', 'Liter', 'Make_ord']]
        x = sm.add_constant(x1)
        y = data_frame['Price']
        est = sm.OLS(y, x).fit()
        print est.predict(x[:20])
        print est.summary();

MultiVariate()
#LinearRegression()
#PolynomialRegression()
