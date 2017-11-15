# Authors:
#   Nicholas Alexander
#   Samuel Levya
#   Jeff Stanton 16547207
# Team: Zotbots

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def train():
    """
    Extracts data from text files and stores in numpy arrays.
    Model's parameters are optimized using this data.
    Results are saved in a text file for submission to Kaggle.
    """

    # get data here from text file (x_train, y_train) - maybe split training test into 70/30 train/validation set

    # process data? need to determine which models would benefit from this

    # instantiate mode
    ada_boost = AdaBoostClassifier()
    gradient_boost = GradientBoostingClassifier()
    random_forest = RandomForestClassifier()

    # get hyper-parameters
    ada_parameters = get_parameters(type(ada_boost).__name__)
    gradient_parameters = get_parameters(type(gradient_boost).__name__)
    forest_parameters = get_parameters(type(random_forest).__name__)

    # optimize hyper-parameters
    # ada_boost = optimize_parameters(ada_boost, ada_parameters, x, y)
    # gradient_boost = optimize_parameters(gradient_boost, gradient_parameters, x, y)
    # random_forest = optimize_parameters(random_forest, forest_parameters, x, y)

    # use trained model to make predictions on x_test
    # average predictions
    # save avg predictions to text file (I believe the instructions are on the latest discussion notebook)

    pass


def process_data(x, y):
    """
    Could be used to regularize data, add polynomial features, or whatever else we need to do.
    Update docstring once/if we figure it out.

    :param x: features
    :param y: labels
    :return: processed x and y data sets
    """
    pass


def get_parameters(model_name):
    """
    Returns a dictionary containing the specified model's hyper-parameters to be optimized.

    :param model_name: the name of the model/estimator
    :return: a dictionary of hyper-parameters where the <key, value> is <parameter_name, [values]>
    """
    return {"AdaBoostClassifier": {},
            "GradientBoostingClassifier": {},
            "RandomForestClassifier": {}}[model_name]


def optimize_parameters(model, parameters, x, y):
    """
    Optimizes model's hyper-parameters through either GridSearchCV (exhaustive) or RandomizedSearchCV.

    :param model: the estimator/learner to be optimized
    :param parameters: a dictionary containing { parameter : [values] } to be optimized
    :param x: features
    :param y: labels
    :return: the trained model
    """
    search = GridSearchCV(model, param_grid=parameters, verbose=1, )
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
    pass


def save(result):
    """
    Saves the prediction to a text file.

    :param result: an average of all the model's predictions
    """
    pass


if __name__ == "__main__":
    train()
