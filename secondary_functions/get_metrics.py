'''
Вспомогательные функции для получения метрик, их конкатенации в датафрейм и проверки переобучения.
'''


import pandas as pd
import numpy as np
from typing import Any
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def get_metrics(y_true: np.array, y_pred: np.array, X_test: np.array,
                model_name: str) -> pd.DataFrame:
    '''Функция для получения метрик по предсказаниям модели. '''
    def r2_adjusted(y_true: np.ndarray, y_pred: np.ndarray,
                    X_test: np.ndarray) -> float:
        """ Коэффициент детерминации (множественная регрессия). """
        n_objects = len(y_true)
        n_features = X_test.shape[1]
        r2 = r2_score(y_true, y_pred)
        return 1 - (1 - r2) * (n_objects - 1) / (n_objects - n_features - 1)

    def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Weighted Absolute Percent Error. """
        return np.sum(np.abs(y_pred - y_true)) / np.sum(y_true) * 100

    try:
        r2 = r2_adjusted(y_true, y_pred, X_test)
        wape = wape(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        df_metrics = pd.DataFrame({
            'Модель': [model_name],
            'MAE': [mae],
            'R2_Adjusted': [r2],
            'WAPE': [wape],
            'RMSE': [np.sqrt(mean_squared_error(y_true, y_pred))]
        })

    except Exception as err:
        df_metrics = pd.DataFrame({
            'Модель': [model_name],
            'MAE': [err],
            'R2_Adjusted': [err],
            'WAPE': [err],
            'RMSE': [err]
        })

    return df_metrics


def concat_metrics(dataframe: pd.DataFrame, y_true: np.array, y_pred: np.array,
                   X_test: np.array, model_name: str) -> pd.DataFrame:
    '''Функция для добавление метрик новых моделей в датасет с метриками. '''
    dataframe = pd.concat([
        dataframe,
        get_metrics(y_true, y_pred, X_test, model_name=model_name)
    ])

    return dataframe


def check_overfitting(X_train: np.array, y_train: np.array, X_test: np.array,
                      y_test: np.array, model: Any) -> None:
    '''Функция проверки на переобучение. Считает разницу
    в процентах MAE на тренировочной и тестовой выборках. '''
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    if len(train_pred.shape) > 1:
        mae_train = mean_absolute_error(y_train, train_pred[:, 0])
        mae_test = mean_absolute_error(y_test, test_pred[:, 0])
        percent_diff = np.round(abs((mae_test - mae_train) / mae_test * 100), 2)
        print(f'{model.__class__.__name__} \n'
              f'MAE train: {mae_train} \n'
              f'MAE test: {mae_test} \n'
              f'MAE diff: {percent_diff} %')
    else:
        mae_train = mean_absolute_error(y_train, train_pred)
        mae_test = mean_absolute_error(y_test, test_pred)
        percent_diff = np.round(abs((mae_test - mae_train) / mae_test * 100), 2)
        print(f'{model.__class__.__name__} \n'
              f'MAE train: {mae_train} \n'
              f'MAE test: {mae_test} \n'
              f'MAE diff: {percent_diff} %')
