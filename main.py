from bayesian_classifier import Bayesian_classifier

classifier = Bayesian_classifier()

x_train, y_train, x_test, y_test = classifier.load_data('iris.data')
classifier.fit(x_train, y_train)
