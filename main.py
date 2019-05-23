from bayesian_classifier import Bayesian_classifier

classifier = Bayesian_classifier()

classifier.load_data('iris.data')
print(classifier.split_train_test())