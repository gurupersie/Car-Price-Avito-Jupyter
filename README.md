# Car-Price-Avito-Jupyter

Данные получены при помощи парсинга сайта avito.ru. Необходимо создать модель, способную по характеристикам автомобиля и дополнительным параметрам предсказать его цену. 

## Структура: 

data_store - хранилище с данными.
secondary_functions - вспомогательные модули.
1. Parsing.ipynb - парсинг сайта Авито и получение данных.
2. EDA.ipynb - разведочный анализ данных и предобработка данных.
3. Baseline.ipynb - обучение бейзлайнов и выбор окончательной модели.
4. Tuning.ipynb - тюнинг моделей.
5. Evaluating.ipynb - итоговая оценка моделей.
6. Stacking.ipynb - стэкинг.
preprocessing.yml - конфигурационный файл.

## Данные по автомобилям:

        Целевая переменная - Цена.
        
        -  Рейтинг  - рейтинг автомобиля на сайте.
        -  Год выпуска  - год производства автомобиля.
        -  Поколение  - поколение автомобиля.
        -  Пробег, км  - сколько проехал автомобиль в километрах.
        -  История пробега, кол-во записей  - количество записей об истории пробега в автотеке. 
        -  Владельцев по ПТС  - количество владельцев, записанных в ПТС.
        -  Состояние  - общее состояние автомобиля.
        -  Модификация  - модификация автомобиля.
        -  Объём двигателя, л  - объём двигателя в литрах.
        -  Тип двигателя  - тип двигателя.
        -  Коробка передач  - тип коробки передач.
        -  Привод  - тип привода.
        -  Комплектация  - комплектация автомобиля
        -  Тип кузова  - тип кузова автомобиля.
        -  Цвет  - цвет автомобиля.
        -  Руль  - расположение руля.
        -  Управление климатом  - тип системы управления климатом
        -  ПТС  - вид ПТС.
        -  Обмен  - рассматривается ли обмен автомобиля на другой, или альтернативные варианты.
        -  Бренд авто  - наименование бренда авто, производитель.
        -  Модель авто  - наименование модели авто.
        -  Город  - город, в котором продаётся авто.
        -  Регион  - регион, в котором продаётся авто.
        -  Выпуск, кол-во лет  - количество лет, в течение которого шёл выпуск автомобиля
        -  Мощность двигателя, лс  - мощность двигателя автомобиля, в лошадиных силах.
