from bayesian_classifier import Bayesian_classifier


correct_rates = []
datasets = ['iris.data', 'coluna.data', 'dermatology.data', 'artificial.data', 'breast-cancer.data']
# datasets = ['artificial.data']

for dataset in datasets:

    for realization in range(0, 1):

        # print('Realização %d' % realization)

        classifier = Bayesian_classifier()

        x_train, y_train, x_test, y_test = classifier.load_data('Datasets/%s' % dataset)
        classifier.fit(x_train, y_train)

        print('Matriz de confusão para %s' % dataset)
        correct_rate = classifier.evaluate(x_test, y_test, conf_matrix=True)
        correct_rates.append(correct_rate)

        if realization == 0:
            classifier.plot_decision_surface('Datasets/%s' % dataset, 'Datasets/%s' % dataset)

    accuracy = sum(correct_rates)/len(correct_rates)
    std = (sum([(x-accuracy)**2 for x in correct_rates])/len(correct_rates))**0.5

    print("O classificador bayesiano puro acertou %.2f" % (accuracy*100))

    print("O desvio padrão foi %.2f" % std)
