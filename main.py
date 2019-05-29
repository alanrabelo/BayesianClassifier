from bayesian_classifier import Bayesian_classifier, DiscriminantType
import numpy as np

# datasets = ['iris.data']
datasets = ['iris.data', 'coluna.data', 'dermatology.data', 'breast-cancer.data', 'artificial.data']
# datasets = ['breast-cancer.data']

for dataset in datasets:

    print('\n\nestamos no dataset %s' % dataset)

    for type in [DiscriminantType.LINEAR, DiscriminantType.QUADRATIC]:

        correct_rates = []

        for realization in range(0, 30):

            # print('Realização %d' % realization)
            classifier = Bayesian_classifier(type=type)

            x_train, y_train, x_test, y_test = classifier.load_data('Datasets/%s' % dataset)
            classifier.fit(x_train, y_train)

            if realization == 0:
                print('Matriz de confusão para %s no discriminante %s' % (dataset, type.value))
                correct_rate = classifier.evaluate(x_test, y_test, conf_matrix=True)
                # Bayesian_classifier(type=type).plot_decision_surface('Datasets/%s' % dataset, dataset.split('.')[0] + ' - ' +type.value, same_covariances=True)

            else:
                correct_rate = classifier.evaluate(x_test, y_test, conf_matrix=False)

            correct_rates.append(correct_rate)



        accuracy = sum(correct_rates)/len(correct_rates)
        std = (sum([(x-accuracy)**2 for x in correct_rates])/len(correct_rates))**0.5

        print("O classificador bayesiano puro acertou %.2f" % (accuracy*100))

        print("O desvio padrão foi %.2f" % std)



# Variando a matriz de covariância

# for dataset in datasets:
#
#     print('estamos no dataset %s' % dataset)
#
#     for type in [DiscriminantType.LINEAR, DiscriminantType.QUADRATIC]:
#     # for type in [DiscriminantType.LINEAR]:
#
#         correct_rates = []
#
#         for realization in range(0, 1):
#
#             # print('Realização %d' % realization)
#             classifier = Bayesian_classifier(type=type)
#
#             x_train, y_train, x_test, y_test = classifier.load_data('Datasets/%s' % dataset)
#             classifier.fit(x_train, y_train)
#
#
#             average_covariance = np.zeros(list(classifier.covariances.values())[0].shape)
#
#             for key, value in enumerate(classifier.covariances):
#                 average_covariance += classifier.covariances[key]
#
#             average_covariance /= len(classifier.covariances.keys())
#
#             for key, value in enumerate(classifier.covariances):
#                 classifier.covariances[key] = average_covariance
#
#             print('Matriz de confusão para %s' % dataset)
#             correct_rate = classifier.evaluate(x_test, y_test, conf_matrix=True)
#             correct_rates.append(correct_rate)
#
#             if realization == 0:
#
#                 Bayesian_classifier(type=type).plot_decision_surface('Datasets/%s' % dataset, dataset.split('.')[0] + ' - ' +type.value, folder_to_save='Cov/')
#
#
#         accuracy = sum(correct_rates)/len(correct_rates)
#         std = (sum([(x-accuracy)**2 for x in correct_rates])/len(correct_rates))**0.5
#
#         print("O classificador bayesiano puro acertou %.2f" % (accuracy*100))
#
#         print("O desvio padrão foi %.2f" % std)
