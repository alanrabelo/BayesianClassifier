from bayesian_classifier import Bayesian_classifier, DiscriminantType
import numpy as np
import matplotlib.pyplot as plt

datasets = ['iris.data', 'coluna.data', 'artificial.data']
datasets = ['artificial.data']

def gerar_grafico(x, y):

    X = np.array(x)
    Y = np.array(y)
    inds = X.argsort()
    sorted_Y = Y[inds]
    X.sort()

    plt.plot(X, sorted_Y, '-o')

    plt.title('Gráfico Acurácia x rejeição - Artificial')
    plt.xlabel('Taxa de rejeição x 100%')
    plt.ylabel('Acurácia x 100%')
    plt.show()


# for dataset in datasets:
#
#     print('\n\nestamos no dataset %s' % dataset)
#
#     for type in [DiscriminantType.PURE]:
#
#         Wr = [0.04, 0.12, 0.24, 0.36, 0.48]
#
#         for wr in Wr:
#
#             correct_rates = []
#             rejections = []
#
#             for realization in range(0, 20):
#
#                 # print('Realização %d' % realization)
#                 classifier = Bayesian_classifier(type=type)
#                 classifier.wr = wr
#
#
#                 x_train, y_train, x_test, y_test = classifier.load_data('Datasets/%s' % dataset)
#                 best_treshold = classifier.fit(x_train, y_train)
#
#                 correct_rate, rejection = classifier.evaluate(x_test, y_test, conf_matrix=False, rejection_threshold=best_treshold)
#
#                 correct_rates.append(correct_rate)
#                 rejections.append(rejection)
#
#             gerar_grafico(rejections, correct_rates)
#
#
#
#
#             print()
#
#             accuracy = sum(correct_rates)/len(correct_rates)
#             std = (sum([(x-accuracy)**2 for x in correct_rates])/len(correct_rates))**0.5
#
#             rejection_rate = sum(rejections)/len(rejections)
#             std_rejection = (sum([(x-rejection_rate)**2 for x in rejections])/len(rejections))**0.5
#
#             print("Com o Wr=%.2f O classificador bayesiano puro acertou %.2f rejeitando %.2f com um threshold=%s" % (wr, accuracy*100, rejection_rate, best_treshold))
#
#             print("O desvio padrão foi %.2f, para a rejeição foi de %.2f" % (std, std_rejection))



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

import matplotlib.pyplot as plt

x = [0, 0.03, 0.06]
y = [0.97, 1, 1]

gerar_grafico(x, y)




