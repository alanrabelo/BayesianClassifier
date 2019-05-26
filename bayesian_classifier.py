import random
import numpy as np
from scipy.stats import *
import math
from numpy.linalg import inv, det

class Bayesian_classifier:

    averages = {}
    std = {}
    priors = {}
    covariances = {}

    def load_data(self, filename):

        self.dataset = []

        with open('iris.data') as data:

            categories = {}

            for line in data.readlines():

                if len(line) < 10:
                    continue

                data_line = line.split(',')
                data_empty = []
                X = data_line[:-1]

                for value in X:
                    data_empty.append(float(value))
                y = data_line[-1]

                if y in categories:
                    data_empty.append(float(categories[y]))
                else:
                    value = float(len(categories.keys()))
                    data_empty.append(value)
                    categories[y] = value

                self.dataset.append(data_empty)


            return self.split_train_test()

    def split_train_test(self, percentage=0.8, evaluate=0.1):

        shuffled_dataset = self.dataset
        random.shuffle(shuffled_dataset)

        split_point = round(len(shuffled_dataset)*(1-evaluate) * percentage)
        evaluate_point = round(len(shuffled_dataset)*(1-evaluate))

        train = np.array(shuffled_dataset[:split_point])
        test = np.array(shuffled_dataset[split_point:evaluate_point])
        evaluate = np.array(shuffled_dataset[evaluate_point:])

        return train[:, :-1], train[:, -1], test[:, :-1], test[:, -1], evaluate[:, :-1], evaluate[:, -1]

    def fit(self, x_train, y_train):

        values = {}
        number_of_variables = 0

        for index, x_value in enumerate(x_train):

            number_of_variables = len(x_value)

            y_value = list(y_train)[index]

            if y_value in values.keys():
                values[y_value].append(x_value)
            else:
                values[y_value] = [x_value]

        averages = {}

        for key, value in enumerate(values):

            # Cálculo das médias
            all_values = values[key]
            average = sum(all_values)/len(all_values)
            self.averages[key] = average

            self.priors[key] = len(all_values) / sum([len(w) for w in values.values()])

            # Cálculo das Variâncias
            for value in all_values:
                var_res = sum((x_input - average) ** 2 for x_input in all_values) / len(all_values)
                self.std[key] = var_res ** 0.5

        # x_train = np.array(x_train)

        from itertools import product

        number_of_variables = len(x_train[0])
        self.covariance = np.zeros((number_of_variables, number_of_variables))

        combinations = [p for p in product(list(range(0, number_of_variables)), repeat=2)]

        values_by_class = {}

        for index, value in enumerate(x_train):

            if y_train[index] in values_by_class:
                values_by_class[y_train[index]].append(value)
            else:
                values_by_class[y_train[index]] = [value]



        for key in values_by_class:

            values = np.array(values_by_class[key])

            covariance_matrix = np.zeros((number_of_variables, number_of_variables))

            for x1, x2 in combinations:

                var_x1_values = values[:, x1]
                var_x2_values = values[:, x2]
                x1_average = sum(var_x1_values)/len(var_x1_values)
                x2_average = sum(var_x2_values)/len(var_x2_values)


                covariance = (sum(var_x1_values * var_x2_values) / len(var_x1_values)) - (x1_average * x2_average)
                covariance_matrix[x1][x2] = covariance
            self.covariances[key] = covariance_matrix

        # cálculo da matriz de covariância


        # print("Averages will be here: \n %s" % str(self.averages))
        # print("Priors will be here: \n %s" % str(self.priors))
        # print("Variances will be here: \n %s" % str(self.std))
        # print("CoVariances will be here: \n %s" % str(self.covariance))


    # def prior(self, train_y):

    def gaussian(self, x_test, cov, u1, pw=0.5):
        u1_transp = u1.transpose()
        x_test_transp = x_test.transpose()

        return -0.5 * (np.dot(np.dot(x_test_transp, inv(cov)), x_test)) + \
               0.5 * (np.dot(np.dot(x_test_transp, inv(cov)), u1)) + \
               0.5 * (np.dot(np.dot(u1_transp, inv(cov)), x_test)) - \
               0.5 * (np.dot(np.dot(u1_transp, inv(cov)), u1)) + math.log(pw) \
               - (0.5 * math.log(2 * math.pi)) \
               - (0.5 * math.log(det(cov)))

    def evaluate(self, x_test, y_test):

        number_of_elements = float(len(x_test))
        acertos = 0
        erros = 0

        for index,value in enumerate(x_test):

            y = self.predict(value)

            if y == y_test[index]:
                acertos += 1
            else:
                erros += 1

        return acertos / number_of_elements


    def predict(self, x):

        max_probability = -math.inf
        best_classe = 0

        for possible_class in self.averages.keys():

            probability = self.gaussian(x, self.covariances[possible_class], self.averages[possible_class], self.priors[possible_class])

            if probability > max_probability:
                max_probability = probability
                best_classe = possible_class

        return best_classe











