'''
Функция отрисовки графика barplot с отображением процентов на столбцах.
'''

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_bars(df: pd.DataFrame,
              target: str,
              feature: str,
              ax: np.array = None) -> matplotlib.figure.Figure:
    '''Функция отрисовки графиков.
       df: pd.DataFrame - исходный датасет с данными
       target: str - наименовение целевой переменной, по которой группируем
       feature: str - наименование колонки-признака, которую рассматриваем
       ax: np.array - подграфик фигуры
    '''
    normed_groups = df.groupby(target)[feature].value_counts(
        normalize=True).mul(100).rename('percent').reset_index()

    ax = sns.barplot(normed_groups,
                     x=target,
                     y='percent',
                     palette='bright',
                     hue=feature,
                     ax=ax,
                     legend='brief')

    for p in ax.patches[:-df[feature].nunique()]:
        percentage = f'{p.get_height():.1f} %'
        ax.annotate(percentage,
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center',
                    va='center',
                    textcoords='offset points',
                    xytext=(0, 20),
                    rotation=90,
                    fontsize=8)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_title(feature, fontsize=18)
    ax.set_xlabel('Классы', fontsize=14)
    ax.set_ylabel('Проценты', fontsize=14)
    ax.set_ylim(0, 115)
    ax.set_xticklabels(df[target].unique(), rotation=45)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    return ax
