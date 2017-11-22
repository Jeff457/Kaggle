# Authors:
#   Nicholas Alexander
#   Samuel Levya
#   Jeff Stanton 16547207
# Team: Zotbots

import numpy as np
from sklearn import preprocessing

import utilities as utils
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


def train():
    """
    Extracts data from text files and stores in numpy arrays.
    Model's parameters are optimized using this data.
    Results are saved in a text file for submission to Kaggle.
    """
    print("Getting data...")
    x_train = np.genfromtxt('data/X_train.txt', delimiter=None)
    y_train = np.genfromtxt('data/Y_train.txt', delimiter=None)
    x_test = np.genfromtxt('data/X_test.txt', delimiter=None)

    # scale features to unit variance and 0 mean
    x_train_scaled = process_data(x_train)
    x_test_scaled = process_data(x_test)

    # instantiate models
    ada_boost = AdaBoostClassifier()
    gradient_boost = GradientBoostingClassifier()
    random_forest = RandomForestClassifier()

    # get hyper-parameters
    ada_parameters = get_parameters(type(ada_boost).__name__)
    gradient_parameters = get_parameters(type(gradient_boost).__name__)
    forest_parameters = get_parameters(type(random_forest).__name__)

    # optimize hyper-parameters
    # ada_boost = optimize_parameters(ada_boost, ada_parameters, x_train, y_train)
    # gradient_boost = optimize_parameters(gradient_boost, gradient_parameters, x_train, y_train)
    random_forest = optimize_parameters(random_forest, forest_parameters, x_train, y_train)

    # display model name, the scoring model used, the model's score, and it's highest performing parameters
    for model in [random_forest]:
        print("{} {} score = {}".format(model, model.scorer_, model.score(x_train, y_train)))
        print("{} highest scoring parameters: {}".format(model, model.best_params_))

    forest_results = np.vstack((np.arange(x_test.shape[0]), random_forest.predict(x_test))).T

    # predict on test data with trained models
    # ada_results = ada_boost.predict(x_test)
    # gradient_results = gradient_boost.predict(x_test)
    # forest_results = random_forest.predict(x_test)

    # average predictions from all the models
    # avg_results = average_predictions(ada_results, gradient_results, forest_results)
    # submission = np.vstack((np.arange(x_test.shape[0]), avg_results)).T
    # save(submission)


def process_data(x):
    """
    Transforms features into a standard normal distribution (zero mean and unit variance).

    :param x: features
    :return: processed data set
    """
    return preprocessing.scale(x)


def get_parameters(model_name):
    """
    Returns a dictionary containing the specified model's hyper-parameters to be optimized.

    :param model_name: the name of the model/estimator
    :return: a dictionary of hyper-parameters where the <key, value> is <parameter_name, [values]>
    """
    return {"AdaBoostClassifier": {},
            "GradientBoostingClassifier": {},
            "RandomForestClassifier": {
                "n_estimators": utils.gen_params(1, 51, 1),
                "criterion": ["gini", "entropy"],
                "min_samples_split": utils.gen_params(2, 31, 2),
                "min_samples_leaf": utils.gen_params(1, 11, 1),
                "oob_score": [True, False],
                "warm_start": [True, False]
            }}[model_name]


def optimize_parameters(model, parameters, x, y):
    """
    Optimizes model's hyper-parameters through either GridSearchCV (exhaustive) or RandomizedSearchCV.

    :param model: the estimator/learner to be optimized
    :param parameters: a dictionary containing { parameter : [values] } to be optimized
    :param x: features
    :param y: labels
    :return: the trained model
    """
    print("Optimizing hyper-parameters...")
    search = RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=10, scoring='roc_auc',
                                verbose=1, n_jobs=-1)
    search.fit(x, y)

    return search  # return the trained model


def average_predictions(ada_results, gradient_results, forest_results):
    """
    Averages (or by whatever means we decide) the predictions from all 3 models.

    :param ada_results: predictions from AdaBoostClassifier
    :param gradient_results: predictions from GradientBoostingClassifier
    :param forest_results: predictions from RandomForestClassifier
    :return: numpy array of the averaged predictions
    """
    return np.mean(np.array([ada_results, gradient_results, forest_results]), axis=0)


def save(result):
    """
    Saves the prediction to a text file.

    :param result: an average of all the model's predictions
    """
    file_name = "y_submission.txt"
    print("Saving submission to {}".format(file_name))
    np.savetxt(file_name, result, '%d, %.5f', header='ID,Prob1', comments='', delimiter=',')


if __name__ == "__main__":
    train()
