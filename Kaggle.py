# Authors:
#   Nick Jackson
#   Sam Levya
#   Jeff Stanton
# Team: Zotbots

import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

import utilities as utils


def train(load_models=False, get_train_val_score=False):
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

    if not load_models:
        # instantiate models
        ada_boost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=15))
        gradient_boost = GradientBoostingClassifier()
        random_forest = RandomForestClassifier()

        # get hyper-parameters
        ada_parameters = get_parameters(type(ada_boost).__name__)
        gradient_parameters = get_parameters(type(gradient_boost).__name__)
        forest_parameters = get_parameters(type(random_forest).__name__)

        # optimize hyper-parameters
        ada_boost = optimize_parameters(ada_boost, ada_parameters, x_train, y_train)
        gradient_boost = optimize_parameters(gradient_boost, gradient_parameters, x_train, y_train)
        random_forest = optimize_parameters(random_forest, forest_parameters, x_train, y_train)

        # display model name, the scoring model used, the model's score, and it's highest performing parameters
        for model in [gradient_boost]:
            model_score = model.score(x_train, y_train)
            print("{} {} Train score = {}".format(type(model.estimator).__name__, model.scorer_, model_score))
            # print("Validation score = {}".format(validation_score))
            print("{} highest scoring parameters: {}\n".format(type(model.estimator).__name__, model.best_params_))
            save_model(model, model_score)  # save fitted model and score
    else:
        print("Loading saved models...")
        ada_boost = get_model("AdaBoostClassifier")
        gradient_boost = get_model("GradientBoostingClassifier")
        random_forest = get_model("RandomForestClassifier")

    if get_train_val_score:
        train_validation_score(x_train, y_train, [random_forest, ada_boost, gradient_boost])

    # predict on test data with trained models
    print("Making predictions...")
    ada_results = ada_boost.predict_proba(x_test)
    gradient_results = gradient_boost.predict_proba(x_test)
    forest_results = random_forest.predict_proba(x_test)

    # average predictions from all the models
    avg_results = average_predictions([forest_results, ada_results, gradient_results], [0.4, 0.2, 0.4])
    submission = np.vstack((np.arange(x_test.shape[0]), avg_results[:, 1])).T
    save(submission)


def train_validation_score(x_train, y_train, models):
    """
    Determines the average train and validation ROC AUC score for a given list of models.

    :param x_train: given data set to split
    :param y_train: given data set to split
    :param models: a list of models to make predictions from
    """
    # to determine train and validation scores for the report
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

    train_results = list()
    validation_results = list()
    for i, model in enumerate(models):
        train_results.append(model.predict_proba(x_train))
        validation_results.append(model.predict_proba(x_val))
        print("Model = {}, Train = {}, Validation = {}".format(type(model.estimator).__name__,
                                                               roc_auc_score(y_train, train_results[i][:, 1]),
                                                               roc_auc_score(y_val, validation_results[i][:, 1])))

    avg_y_train = average_predictions(train_results)[:, 1]
    avg_y_val = average_predictions(validation_results)[:, 1]

    print("Train Score = {}, Validation Score = {}".format(roc_auc_score(y_train, avg_y_train),
                                                           roc_auc_score(y_val, avg_y_val)))


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
    return {
            "AdaBoostClassifier": {
                # "base_estimator": [],
                # base_estimator = DecisionTreeClassifier
                "base_estimator__criterion": ["gini", "entropy"],
                "base_estimator__min_samples_split": utils.gen_params(2, 31, 2),
                "base_estimator__min_samples_leaf": utils.gen_params(1, 21, 2),
                "base_estimator__max_features": ["sqrt", "log2", None],
                "base_estimator__presort": [True, False],
                "n_estimators": utils.gen_params(500, 1000, 50),
                "learning_rate": utils.gen_params(0.001, 0.1, 0.001),
                "algorithm": ["SAMME.R"]
            },
            "GradientBoostingClassifier": {
                "loss": ['deviance', 'exponential'],
                "n_estimators": utils.gen_params(40, 201, 10),
                "learning_rate": utils.gen_params(0.4, 1.2, 0.1),
                "min_samples_split": utils.gen_params(7, 201, 2),
                "min_samples_leaf": utils.gen_params(3, 201, 1),
                "max_features": ["sqrt", "log2", None],
                "max_depth": utils.gen_params(4, 20, 1)
            },
            "RandomForestClassifier": {
                "n_estimators": utils.gen_params(100, 501, 10),
                "criterion": ["gini", "entropy"],
                "max_features": ["sqrt", "log2", None],
                "min_samples_split": utils.gen_params(2, 31, 2),
                "min_samples_leaf": utils.gen_params(1, 21, 1),
                "oob_score": [True, False],
                "warm_start": [True, False]
            }
            }[model_name]


def optimize_parameters(model, parameters, x, y):
    """
    Optimizes model's hyper-parameters through either GridSearchCV (exhaustive) or RandomizedSearchCV.

    :param model: the estimator/learner to be optimized
    :param parameters: a dictionary containing { parameter : [values] } to be optimized
    :param x: features
    :param y: labels
    :return: the trained model
    """
    print("Optimizing hyper-parameters for {}...".format(type(model).__name__))
    search = RandomizedSearchCV(estimator=model, param_distributions=parameters, cv=5, scoring='roc_auc',
                                n_iter=100, verbose=10, n_jobs=-1)
    search.fit(x, y)

    return search  # return the trained model


def save_model(model, model_score):
    """
    Checks if the model's current score is higher than its previous score.  If so, overwrites previous model.

    :param model: the model to persist
    :param model_score: the model's roc_auc score
    """
    score_file_name = type(model.estimator).__name__ + "Score.txt"
    score_file = utils.get_file_path(score_file_name)

    model_file_name = type(model.estimator).__name__ + ".pkl"
    overwrite = False

    # check if we've already persisted this model.  if so, check if new model scored higher
    if score_file.exists():
        with open(score_file_name, 'r') as file:
            prev_score = float(file.read())
        print("Previous score = {}, new score = {}".format(prev_score, model_score))
        overwrite = prev_score < model_score

    if overwrite or not score_file.exists():
        print("Saving {}...".format(type(model.estimator).__name__))
        joblib.dump(model, model_file_name)
        with open(score_file_name, 'w') as file:
            file.write(str(model_score))


def get_model(model):
    """
    Gets the specified persisted model.

    :param model: the model to retrieve
    :return: a fitted estimator
    """
    model_file_name = model + ".pkl"
    model_file = utils.get_file_path(model_file_name)

    if model_file.exists():
        return joblib.load(model_file_name)
    return None


def average_predictions(iterable, weights):
    """
        Averages (or by whatever means we decide) the predictions from all 3 models.

        :param iterable: a list of ndarray predictions
        :return: numpy array of the averaged predictions
        """
    print("Averaging results...")
    return np.average(np.array(iterable), axis=0, weights=weights)


def save(result):
    """
    Saves the prediction to a text file.

    :param result: an average of all the model's predictions
    """
    file_name = "y_submission.txt"
    print("Saving submission to {}".format(file_name))
    np.savetxt(file_name, result, '%d, %.5f', header='ID,Prob1', comments='', delimiter=',')


if __name__ == "__main__":
    train(load_models=True, get_train_val_score=False)
