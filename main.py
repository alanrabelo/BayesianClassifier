from bayesian_classifier import Bayesian_classifier

classifier = Bayesian_classifier()

x_train, y_train, x_test, y_test, x_evaluation, y_evaluation = classifier.load_data('iris.data')
classifier.fit(x_train, y_train)

accuracy = classifier.evaluate(x_test, y_test)

print("O classificador bayesiano acertou %.2f%%" % (accuracy * 100))