import random
import numpy as np

class Bayesian_classifier:

    averages = {}
    std = {}

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

    def split_train_test(self, percentage=0.8):

        shuffled_dataset = self.dataset
        random.shuffle(shuffled_dataset)

        split_point = round(len(shuffled_dataset)*percentage)

        train = np.array(shuffled_dataset[:split_point])
        test = np.array(shuffled_dataset[split_point:])

        return train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

    def fit(self, x_train, y_train):

        values = {}

        for index, x_value in enumerate(x_train):

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

            # Cálculo das Variâncias
            for value in all_values:
                var_res = sum((x_input - average) ** 2 for x_input in all_values) / len(all_values)
                self.std[key] = var_res ** 0.5


        # cálculo da matriz de covariância

        print("Averages will be here: \n %s" % str(self.averages))

        print("Variances will be here: \n %s" % str(self.std))










