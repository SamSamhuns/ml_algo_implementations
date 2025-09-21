import numpy as np
import logging
import sys
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
plt.style.use("seaborn-darkgrid")

model_parameter = namedtuple("model_parameter", ("lambda_ridge, alpha, epochs"))


class Preprocessing:
    def __init__(self):
        self.mean = None
        self.std = None

    @staticmethod
    def standardize(X, mean=None, std=None, inplace=False):
        if mean is None:
            mean = np.mean(X, axis=0)
        if std is None:
            std = np.std(X, axis=0)

        std = np.where(std == 0, 1, std)
        if inplace:
            X -= mean
            X /= std
        else:
            X = (X - mean) / std
        return X

    @staticmethod
    def insert_bias_term(X):
        bias_arr = np.ones(X.shape[0])
        return np.c_[bias_arr, X]

    def standardize_save_state(self, X, mean=None, std=None, inplace=False):
        if mean is None:
            mean = np.mean(X, axis=0)
        if std is None:
            std = np.std(X, axis=0)

        std = np.where(std == 0, 1, std)
        self.mean = mean
        self.std = std
        if inplace:
            X -= mean
            X /= std
        else:
            X = (X - mean) / std
        return X

    def fit(self, X, inplace=False):
        if self.mean is None or self.std is None:
            raise ValueError("Mean or std is not for the preprocessing object")
        if inplace:
            X -= self.mean
            X /= self.std
        else:
            X = (X - self.mean) / self.std
        return X


class LogisticRegr:
    slots = ["theta"]

    def __init__(self):
        self.theta = None

    def __repr__(self):
        return " ".join([str(val) for val in np.ndarray.flatten(self.theta)])

    def fit(
        self,
        X,
        y,
        logging_enabled=True,
        model_params=model_parameter(lambda_ridge=0, alpha=0.5, epochs=5000),
    ):
        """
        WARNING: X must be normalized and have the bias term before training
        Batch Gradient Descent for logistic regression
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X shape {X.shape[0]} != y shape {y.shape[0]}. Dimensions not matching"
            )

        loss_arr = []
        m = X.shape[0]
        self.theta = np.zeros(X.shape[1])
        lambda_ridge, alpha, epochs = model_params

        for epoch in range(epochs):
            gradient_wout_regu = (1 / m) * np.dot(
                np.matrix.transpose(X), LogisticRegr.sigmoid(np.dot(X, self.theta)) - y
            )
            # 0th parameter/bias is not regularized
            self.theta[0] = self.theta[0] - alpha * gradient_wout_regu[0]
            gradient_with_regu = gradient_wout_regu + ((lambda_ridge / m) * self.theta)
            # All other parameters regularized
            self.theta[1:] = self.theta[1:] - alpha * gradient_with_regu[1:]

            if epoch % 100 == 0:
                current_log_loss = self.loss(X, y, lambda_ridge)
                if logging_enabled:
                    print(f"Loss at epoch {epoch} is {current_log_loss}")
                loss_arr.append(current_log_loss)

        if logging_enabled:
            self.plot_loss_curve(loss_arr, epochs)

    def plot_loss_curve(self, loss_arr, epochs, log_scale: bool = False):
        if log_scale:
            plt.semilogx(range(epochs), loss_arr)
        else:
            plt.plot(loss_arr)
        plt.ylabel("log loss")
        plt.xlabel("Epoch (x100)")
        plt.title("Loss Overtime")
        plt.grid(True)
        plt.show()

    @staticmethod
    def log_loss(X, y, theta, lambda_ridge: float = 0.0):
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X shape {X.shape[0]} != y shape {y.shape[0]}. Dimensions not matching"
            )
        elif X.shape[1] != theta.shape[0]:
            raise ValueError(
                f"X shape {X.shape[1]} != theta shape {theta.shape[0]}. Dimensions not matching"
            )

        m = X.shape[0]
        h = LogisticRegr.sigmoid(np.dot(X, theta))
        # loss J(theta) = -(1/m)*(yt*logh + (1-y)t*log(1-h)) + lambda/2m theta_t * theta
        return (-1 / m) * (
            np.dot(np.matrix.transpose(y), np.log(h))
            + np.dot(np.matrix.transpose(1 - y), np.log(1 - h))
        ) + (
            (lambda_ridge / (2 * m)) * np.dot(np.matrix.transpose(theta[1:]), theta[1:])
        )

    @staticmethod
    def sigmoid(X):
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def predict(X, theta, threshold: float = 0.5):
        prediction = np.dot(X, theta)
        prediction[prediction >= threshold] = 1
        prediction[prediction < threshold] = 0
        return prediction

    def loss(self, X, y, lambda_ridge: float = 0.0):
        return LogisticRegr.log_loss(X, y, self.theta, lambda_ridge)

    def accuracy(self, X, y, threshold: float = 0.5):
        """
        accuracy = (TP+TN) / (TP+FP+TN+FN)
        """
        y_pred = LogisticRegr.predict(X, self.theta, threshold)
        return (
            len([1 for y_true, y_hat in zip(y, y_pred) if y_true == y_hat]) / X.shape[0]
        )

    def precision(self, X, y, threshold: float = 0.5):
        """
        Ratio of correctly predicted positive observations to total positive observations
        precision = TP / (TP+FP)
        """
        y_pred = LogisticRegr.predict(X, self.theta, threshold)
        true_positives = len(
            [1 for y_true, y_hat in zip(y, y_pred) if y_true == y_hat == 1]
        )
        total_positives_pred = sum(y_pred)
        return true_positives / total_positives_pred

    def recall(self, X, y, threshold: float = 0.5):
        """
        Also known as Sensitivity
        Ratio of correctly predicted positive observations to all observations that are actually positive
        recall = TP / (TP+FN)
        """
        y_pred = LogisticRegr.predict(X, self.theta, threshold)
        true_positives = len(
            [1 for y_true, y_hat in zip(y, y_pred) if y_true == y_hat == 1]
        )
        true_pos_and_false_neg = len(
            [
                1
                for y_true, y_hat in zip(y, y_pred)
                if y_true == y_hat == 1 or (y_true == 1 and y_hat == 0)
            ]
        )
        return true_positives / true_pos_and_false_neg

    def f1_score(self, X, y, threshold: float = 0.5):
        """
        Weighted average of precision and recall
        Preferred to accuracy as accuracy is misleading for unbalanced datasets
        f1_score = 2*(Recall*Precision) / (Recall+Precision)
        """
        recall = self.recall(X, y, threshold)
        precision = self.precision(X, y, threshold)
        return (2 * recall * precision) / (recall + precision)

    def plot_confusion_matrix(self, X, y, threshold: float = 0.5, custom: bool = True):
        y_pred = LogisticRegr.predict(X, self.theta, threshold)

        tp = len([1 for y_true, y_hat in zip(y, y_pred) if y_true == y_hat == 1])
        tn = len([1 for y_true, y_hat in zip(y, y_pred) if y_true == y_hat == 0])
        fp = len([1 for y_true, y_hat in zip(y, y_pred) if y_true == 0 and y_hat == 1])
        fn = len([1 for y_true, y_hat in zip(y, y_pred) if y_true == 1 and y_hat == 0])
        if custom:  # use custom confusion matrix generator
            print("\t\t\t      Actual values")
            print("\t\t\tPositive(1)   Negative(0)")
            print(f"Predicted| Positive(1)     TP {tp}\t  FP {fp}")
            print(f"  Values | Negative(0)     FN {fn}\t\t  TN {tn}")
        else:  # use sklearn.metrics.confusion_matrix
            pass


class KFoldCrossValidator:
    __slots__ = ["train_loss", "test_loss", "train_accuracy", "test_accuracy", "theta"]

    def __init__(self):
        self.train_loss = []
        self.test_loss = []
        self.train_accuracy = []
        self.test_accuracy = []
        self.theta = None

    def cross_validate(
        self,
        model,
        X,
        y,
        k=10,
        logging_enabled=True,
        model_params=model_parameter(lambda_ridge=0, alpha=0.5, epochs=5000),
        custom_kfold=False,
        seed=np.random.randint(10000),
    ):
        """
        Cross validation function, the theta parameter chosen is from the split with the least test error
        """

        m = X.shape[0]
        lambda_ridge, alpha, epochs = model_params
        min_test_error = float("inf")  # tracks the minimum error with k-folds
        best_fit_theta = None  # saves the best theta value with the min_test_error
        preprocessor_object = Preprocessing()

        if custom_kfold:
            logging.info(
                f"Running Custom KFoldCrossValidator with {k} folds and lambda={lambda_ridge}"
            )
            np.random.seed(seed)  # seed random shuffler
            if m < k:
                raise ValueError(
                    f"No of k splits {k} cannot be greater than no. of samples {m}"
                )

            # Randomly shuffle X and y inplace while matching corresponding feat and target
            for i in range(m):
                swap_idx = np.random.randint(i, m)
                # ensures the corresponding feat-target values match
                X[[i, swap_idx]] = X[[swap_idx, i]]
                y[[i, swap_idx]] = y[[swap_idx, i]]

            # test start and end idx
            fold_step = m // k
            start = 0
            end = fold_step

            for i in range(k):
                end = min(end, m)  # prevent array idx out of bounds
                X_train, X_test = (
                    np.concatenate([X[0:start], X[end:m]], axis=0),
                    X[start:end],
                )
                y_train, y_test = (
                    np.concatenate([y[0:start], y[end:m]], axis=0),
                    y[start:end],
                )
                start += fold_step
                end += fold_step

                X_train = preprocessor_object.standardize_save_state(X_train)
                # standardizing X_test with X_train params
                X_test = preprocessor_object.fit(X_test)

                X_train = Preprocessing.insert_bias_term(X_train)
                X_test = Preprocessing.insert_bias_term(X_test)

                model.fit(X_train, y_train, logging_enabled, model_params)
                cur_train_loss = model.loss(X_train, y_train, lambda_ridge)
                cur_test_loss = model.loss(X_test, y_test, lambda_ridge)
                self.train_loss.append(cur_train_loss)
                self.test_loss.append(cur_test_loss)

                if cur_test_loss < min_test_error:
                    min_test_error = cur_test_loss
                    best_fit_theta = model.theta
        else:
            logging.info(
                f"Running Sklearn KFoldCrossValidator with {k} folds and lambda {lambda_ridge}"
            )
            kf = KFold(n_splits=k, random_state=seed, shuffle=True)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                X_train = preprocessor_object.standardize_save_state(X_train)
                # standardizing X_test with X_train params
                X_test = preprocessor_object.fit(X_test)

                X_train = Preprocessing.insert_bias_term(X_train)
                X_test = Preprocessing.insert_bias_term(X_test)

                model.fit(X_train, y_train, logging_enabled, model_params)
                cur_train_loss = model.loss(X_train, y_train, lambda_ridge)
                cur_test_loss = model.loss(X_test, y_test, lambda_ridge)
                self.train_loss.append(cur_train_loss)
                self.test_loss.append(cur_test_loss)

                if cur_test_loss < min_test_error:
                    min_test_error = cur_test_loss
                    best_fit_theta = model.theta
        self.theta = best_fit_theta


if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.40, random_state=42
    )

    preprocessing_object = Preprocessing()
    X_train_std = preprocessing_object.standardize_save_state(X_train)
    X_train_std = preprocessing_object.insert_bias_term(X_train_std)

    bcw_model = LogisticRegr()
    model_params = model_parameter(lambda_ridge=0, alpha=0.5, epochs=5000)

    bcw_model.fit(X_train_std, y_train, model_params)
