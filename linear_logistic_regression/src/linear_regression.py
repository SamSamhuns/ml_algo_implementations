import numpy as np
import logging
import sys
from enum import Enum
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold

logging.basicConfig(stream=sys.stderr, level=logging.INFO)


class TrainType(Enum):
    NORMAL_EQUATION = 0
    GRADIENT_DESCENT = 1


Model_Parameter = namedtuple(
    "Model_Parameter", ("lambda_ridge, training_method, alpha, epochs")
)


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


class LinearRegr:
    __slots__ = ["theta"]

    def __init__(self):
        self.theta = None

    def __repr__(self):
        return " ".join([str(parm) for parm in np.ndarray.flatten(self.theta)])

    def fit(
        self,
        X,
        y,
        model_params=Model_Parameter(
            lambda_ridge=0,
            training_method=TrainType.NORMAL_EQUATION,
            alpha=0.5,
            epochs=1000,
        ),
    ):
        """
        Fit/train the linear model
        It has been assumed that the bias has been added to X
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X shape {X.shape[0]} != y shape {y.shape[0]}. Dimensions not matching"
            )

        if model_params.training_method == TrainType.NORMAL_EQUATION:
            self._normal_equation_method(X, y, model_params)
        elif model_params.training_method == TrainType.GRADIENT_DESCENT:
            self._gradient_descent_method(X, y, model_params)
        else:
            raise ValueError("Model type not supplied")

    def _normal_equation_method(self, X, y, model_params):
        # Feature Scaling is not required
        # theta = (XtX + LE*)-1 . Xt.y
        # Almost identity matrix E where the first row, first col elem is 0
        # since we do not regularize the bias input, x0 = 1
        lambda_ridge = model_params.lambda_ridge

        E_start = np.identity(X.shape[1])
        E_start[0][0] = 0
        E_start *= lambda_ridge
        X_t = np.matrix.transpose(X)

        dot_Xt_X = np.dot(X_t, X)  # XtX
        self.theta = np.dot(np.dot(np.linalg.pinv(dot_Xt_X + E_start), X_t), y)

    def _gradient_descent_method(self, X, y, model_params):
        """
        WARNING Feature scaling should already be done for X
        Batch Gradient Descent
        """
        lambda_ridge, training_method, alpha, epochs = model_params
        self.theta = np.zeros(X.shape[1])
        loss_overtime = []
        m = y.shape[0]

        for epoch in range(epochs):
            gradient_wout_regu = (1 / m) * np.dot(
                np.matrix.transpose(X), np.dot(X, self.theta) - y
            )
            # 0th parameter/bias is not regularized
            self.theta[0] = self.theta[0] - alpha * gradient_wout_regu[0]
            gradient_with_regu = gradient_wout_regu + ((lambda_ridge / m) * self.theta)
            # All other parameters regularized
            self.theta[1:] = self.theta[1:] - alpha * gradient_with_regu[1:]

            if epoch % 1 == 0:
                current_loss = self.loss(X, y, lambda_ridge)
                logging.info(f"Current loss at epoch {epoch} is {current_loss}")
                loss_overtime.append(current_loss)

        self.plot_loss_curve(loss_overtime, epochs)

    def plot_loss_curve(self, loss_arr, iterations, log_mode=False):
        if log_mode:
            plt.semilogx(range(iterations), loss_arr)
        else:
            plt.plot(loss_arr)
        plt.title("Loss function")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

    @staticmethod
    def mse_loss(X, y, theta, lambda_ridge=0):
        """Calculates the MSE loss for linear regression"""
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X shape {X.shape[0]} != y shape {y.shape[0]}. Dimensions not matching"
            )
        elif X.shape[1] != theta.shape[0]:
            raise ValueError(
                f"X shape {X.shape[1]} != theta shape {theta.shape[0]}. Dimensions not matching"
            )

        X_theta_minus_y = np.dot(X, theta) - y
        X_theta_minus_y_t = np.matrix.transpose(X_theta_minus_y)

        return (1 / (2 * X.shape[0])) * (
            (np.dot(X_theta_minus_y_t, X_theta_minus_y))
            + (lambda_ridge * np.dot(np.matrix.transpose(theta[1:]), theta[1:]))
        )

    @staticmethod
    def predict_X(X, theta):
        """
        Predict using the linear model
        """
        if theta is None:
            raise ValueError("Model has not been trained yet")

        # prediction = X*theta
        return np.dot(X, theta)

    @staticmethod
    def score_X(X, y, theta):
        """
        Returns the coefficient of determination
        """
        if theta is None:
            raise ValueError("Model has not been trained yet")

        y_mean = np.mean(y)
        y_pred = LinearRegr.predict_X(X, theta)
        ss_total = sum((y - y_mean) ** 2)  # total sum of squares
        ss_res = sum((y - y_pred) ** 2)  # sum of squared residuals

        return 1 - (ss_res / ss_total)

    def loss(self, X, y, lambda_ridge=0):
        """
        Calculates the current loss
        """
        if self.theta is None:
            raise ValueError("Model has not been trained yet")

        return LinearRegr.mse_loss(X, y, self.theta, lambda_ridge)

    def score(self, X, y):
        """
        Returns the coefficient of determination
        """
        return LinearRegr.score_X(X, y, self.theta)

    def predict(self, X):
        """
        Predict using the linear model
        """
        return LinearRegr.predict_X(X, self.theta)


class KFoldCrossValidator:
    __slots__ = ["train_loss", "test_loss", "theta"]

    def __init__(self):
        self.train_loss = []
        self.test_loss = []
        self.theta = None

    def cross_validate(
        self,
        model,
        model_params,
        X,
        y,
        k=10,
        custom_kfold=False,
        seed=np.random.randint(10000),
    ):
        """
        Cross validation function, the theta parameter chosen is from the split with the least test error
        """

        m = X.shape[0]
        lambda_ridge, training_method, alpha, epochs = model_params
        min_test_error = float("inf")  # tracks the minimum error with k-folds
        best_fit_theta = None  # saves the best theta value with the min_test_error

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

                model.fit(X_train, y_train, model_params)
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

                model.fit(X_train, y_train, model_params)
                cur_train_loss = model.loss(X_train, y_train, lambda_ridge)
                cur_test_loss = model.loss(X_test, y_test, lambda_ridge)
                self.train_loss.append(cur_train_loss)
                self.test_loss.append(cur_test_loss)

                if cur_test_loss < min_test_error:
                    min_test_error = cur_test_loss
                    best_fit_theta = model.theta
        self.theta = best_fit_theta


if __name__ == "__main__":
    X, y = load_boston(return_X_y=True)
    bh_model = LinearRegr()
    kfold_linear_regr = KFoldCrossValidator()
    X_feat = Preprocessing.insert_bias_term(X)

    model_params = Model_Parameter(
        lambda_ridge=0, training_method=TrainType.NORMAL_EQUATION, alpha=0, epochs=0
    )
    kfold_linear_regr.cross_validate(
        bh_model, model_params, X_feat, y, k=10, custom_kfold=False
    )

    lregr_train_loss = kfold_linear_regr.train_loss
    lregr_test_loss = kfold_linear_regr.test_loss
    print(f"Average train loss is: {sum(lregr_train_loss) / len(lregr_train_loss)}")
    print(f"Average test loss is: {sum(lregr_test_loss) / len(lregr_test_loss)}")

    print(f"R squared for the entire dataset is {bh_model.score(X_feat, y)}")
