import random
import numpy as np
from scipy.stats import *
import math
from numpy.linalg import inv, det
import matplotlib.pyplot as plt

class Bayesian_classifier:

    averages = {}
    std = {}
    priors = {}
    covariances = {}

    def load_data(self, filename, add_evaluation_values=False):

        self.dataset = []

        with open(filename) as data:

            categories = {}

            for line in data.readlines():

                if len(line) < 5 or "?" in line:
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


            if 'breast' in filename:
                self.dataset = np.array(self.dataset)[:, 1:]

            normalized_dataset = np.array(self.dataset, dtype=float)

            number_of_columns = len(self.dataset[0])
            for column in range(number_of_columns - 1):
                current_column = normalized_dataset[:, column]
                max_of_column = float(max(current_column))
                min_of_column = float(min(current_column))

                for index in range(len(self.dataset)):
                    value = self.dataset[index][column]
                    self.dataset[index][column] = (float(value) - min_of_column) / (max_of_column - min_of_column)

            return self.split_train_test_evaluate() if add_evaluation_values else self.split_train_test()

    def split_train_test_evaluate(self, percentage=0.9, evaluate=0.1):

        shuffled_dataset = self.dataset
        random.shuffle(shuffled_dataset)

        split_point = round(len(shuffled_dataset)*(1-evaluate) * percentage)
        evaluate_point = round(len(shuffled_dataset)*(1-evaluate))

        train = np.array(shuffled_dataset[:split_point])
        test = np.array(shuffled_dataset[split_point:evaluate_point])
        evaluate = np.array(shuffled_dataset[evaluate_point:])

        return train[:, :-1], train[:, -1], test[:, :-1], test[:, -1], evaluate[:, :-1], evaluate[:, -1]

    def split_train_test(self, percentage=0.8):

        shuffled_dataset = self.dataset
        random.shuffle(shuffled_dataset)

        split_point = round(len(shuffled_dataset) * percentage)

        train = np.array(shuffled_dataset[:split_point])
        test = np.array(shuffled_dataset[split_point:])

        return train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

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



        equal_covariance = []
        for key in values_by_class:

            values = np.array(values_by_class[key])

            covariance_matrix = np.zeros((number_of_variables, number_of_variables), dtype=float)

            for x1, x2 in combinations:

                var_x1_values = values[:, x1]
                var_x2_values = values[:, x2]
                x1_average = sum(var_x1_values)/len(var_x1_values)
                x2_average = sum(var_x2_values)/len(var_x2_values)


                covariance = (sum(var_x1_values * var_x2_values) / len(var_x1_values)) - (x1_average * x2_average)
                covariance_matrix[x1][x2] = covariance
            equal_covariance = covariance_matrix
            break

        for key in values_by_class:
            self.covariances[key] = equal_covariance


        # cálculo da matriz de covariância


        # print("Averages will be here: \n %s" % str(self.averages))
        # print("Priors will be here: \n %s" % str(self.priors))
        # print("Variances will be here: \n %s" % str(self.std))
        # print("CoVariances will be here: \n %s" % str(self.covariance))


    # def prior(self, train_y):

    # def gaussian(self, x_test, cov, u1, pw=0.5):
    #
    #     u1_transp = u1.transpose()
    #     x_test_transp = x_test.transpose()
    #     det_cov = det(cov)
    #
    #     try:
    #         inv_cov = inv(cov)
    #     except:
    #         inv_cov = cov
    #
    #     result = -0.5 * (np.dot(np.dot(x_test_transp, inv_cov), x_test)) + \
    #            0.5 * (np.dot(np.dot(x_test_transp, inv_cov), u1)) + \
    #            0.5 * (np.dot(np.dot(u1_transp, inv_cov), x_test)) - \
    #            0.5 * (np.dot(np.dot(u1_transp, inv_cov), u1))
    #
    #
    #     try:
    #         log_det_cov = math.log(det_cov)
    #     except:
    #         log_det_cov = 0
    #
    #     return result - log_det_cov - (0.5 * math.log(2 * math.pi)) + math.log(pw)


    def gaussian_linear(self, x_test, cov, u1, pw=0.5):

        u1_transp = u1.transpose()
        x_test_transp = x_test.transpose()
        det_cov = det(cov)

        try:
            inv_cov = inv(cov)
        except:
            inv_cov = cov

        result = 0.5 * (np.dot(np.dot(x_test_transp, inv_cov), u1)) + \
                 0.5 * (np.dot(np.dot(u1_transp, inv_cov), x_test)) \
                -0.5 * (np.dot(np.dot(u1_transp, inv_cov), u1)) \
                 + math.log(pw) - (0.5 * (np.dot(np.dot(u1_transp, inv_cov), u1)))

        return result



    def evaluate(self, x_test, y_test, conf_matrix=False):

        number_of_elements = float(len(x_test))
        acertos = 0
        erros = 0
        confusion_matrix = {}

        for index,value in enumerate(x_test):

            y = self.predict(value)
            desired = int(y_test[index])

            if desired in confusion_matrix:
                if y in confusion_matrix[desired]:
                    confusion_matrix[desired][y] += 1
                else:
                    confusion_matrix[desired][y] = 1
            else:
                confusion_matrix[desired] = {y: 1}

            if y == y_test[index]:
                acertos += 1
            else:
                erros += 1

        if conf_matrix:
            print(confusion_matrix)

        return acertos / number_of_elements


    def predict(self, x):

        max_probability = -math.inf
        best_classe = 0

        for possible_class in self.averages.keys():

            cov = self.covariances[possible_class]
            if len(cov) == 0:
                continue

            probability = self.gaussian_linear(x, self.covariances[possible_class], self.averages[possible_class], self.priors[possible_class])

            if probability > max_probability:
                max_probability = probability
                best_classe = possible_class

        return best_classe


    def plot_decision_surface(self, filename, title):

        # parameter_combination = list(itertools.combinations(range(len(list(X[0]))), 2))

        x_train, y_train, x_test, y_test = self.load_data(filename)

        x_train = x_train[:, :2]
        x_test = x_test[:, :2]

        self.fit(x_train, y_train)

        clear_red = "#ffcccc"
        clear_blue = "#ccffff"
        clear_green = "#ccffcc"
        clear_yellow = "#F5FAA9"
        clear_pink = "#F9A9FA"
        clear_orange = "#FBECD1"

        colors = [clear_red, clear_blue, clear_green, clear_yellow, clear_pink, clear_orange]
        strong_colors = ['red', 'blue', '#2ECC71', '#F9FF2D', '#FF2DF2', '#FFAE00']
        number_of_points = 100

        for i in range(0, number_of_points+1, 1):
            for j in range(0, number_of_points+1, 1):
                x = i / number_of_points
                y = j / number_of_points
                value = int(self.predict(np.array([x, y])))

                color = colors[value]

                plt.plot([x], [y], 'ro', color=color)

        for index, input in enumerate(x_test):
            color_value = int(self.predict(input))
            plt.plot(input[0], input[1], 'ro', color=strong_colors[color_value])

        medium_colors = ['#641E16', '#1B4F72', '#186A3B', '#AAB203', '#970297', '#974F02']

        for index, input in enumerate(x_train):
            color_value = int(self.predict(input))
            plt.plot(input[0], input[1], 'ro', color=medium_colors[color_value])

        plt.suptitle(title)
        plt.show()









