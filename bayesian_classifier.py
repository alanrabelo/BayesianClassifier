import random

class Bayesian_classifier:

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

            return self.dataset

    def split_train_test(self, percentage=0.8):

        shuffled_dataset = self.dataset
        random.shuffle(shuffled_dataset)

        split_point = round(len(shuffled_dataset)*percentage)

        train = shuffled_dataset[:split_point]
        test = shuffled_dataset[split_point:]

        return train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

