import numpy as np
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV



# Первый набор данных
# data1.csv

task1 = pd.read_csv('table.csv', index_col='id')

# Координаты для заданий
x = 52
y = 87
task1['euclid'] = task1.apply(lambda row: sqrt((row['X'] - x)**2 + (row['Y'] - y)**2), axis=1)
task1['manhatton'] = task1.apply(lambda row: abs(row['X'] - x) + abs(row['Y'] - y), axis=1)

task1.sort_values(by='euclid', inplace=True)

print('1.1:',task1.head(1)['euclid'].values[0])

print('1.2:', ','.join(map(str,task1.head(3).reset_index()['id'])))

print('1.3', ','.join(map(str,task1.head(3)['Class'])), '(Выберите, какого класса больше)')

task1.sort_values(by='manhatton', inplace=True)

print('1.4:',task1.head(1)['manhatton'].values[0])

print('1.5:', ','.join(map(str,task1.head(3).reset_index()['id'])))

print('1.6', ','.join(map(str,task1.head(3)['Class'])), '(Выберите, какого класса больше)')

# Второй набор данных
# data2.csv

task2 = pd.read_csv('adult_data_train(2).csv')

# Избавьтесь от признаков education и marital-status.
# Удалите соответствующие колонки из набора данных. Определите количество числовых и нечисловых признаков.
to_delete = ['education', 'marital-status']
task2.drop(to_delete, axis=1, inplace=True)
print('2.1:')
print(task2.dtypes.value_counts())
print('2.2:', task2['label'].value_counts().loc[0] / len(task2))

# 1. Построение базовой модели

# Отберите из набора данных только числовые признаки.
non_numeric_columns = [col for col in task2.columns if task2.dtypes.loc[col] == 'object']
task2_clean = task2.drop(non_numeric_columns, axis=1)


# При помощи train_test_split() разбейте набор данных на обучающую
# и тестовую выборки с параметрами test_size=0.2, random_state=15.
# Используйте стратификацию по колонке label.

x_train, x_test, y_train, y_test = train_test_split(
    task2_clean.drop('label', axis=1),
    task2_clean['label'],
    test_size=0.2,
    random_state=15,
    stratify=task2_clean['label']
)
print('2.3:', x_train['fnlwgt'].mean())
knn1 = KNeighborsClassifier()
knn1.fit(x_train, y_train)
y_pred = knn1.predict(x_test)
print('2.4:', f1_score(y_test, y_pred))

# Масштабирование признаков.

scaler = MinMaxScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

print('2.5:', x_train['fnlwgt'].mean())

knn2 = KNeighborsClassifier()
knn2.fit(x_train, y_train)
y_pred = knn2.predict(x_test)

print('2.6:', f1_score(y_test, y_pred))

print('2.7: Откроются новые окна для просмотра, выберите соответствующие')

fig, axs = plt.subplots(1, len(non_numeric_columns))

for ax, col in zip(axs, non_numeric_columns):
    c = task2[col].value_counts()
    ax.bar(c.index, c)
    ax.set_title(col)

plt.show()

rows_with_nan = task2.apply(lambda row: (row=='?').any(), axis=1)
print('2.8:', rows_with_nan.sum())
task2_with_dummies = task2[~rows_with_nan].copy()

for col in non_numeric_columns:
    dummies = pd.get_dummies(task2_with_dummies[col], drop_first=True)
    task2_with_dummies[dummies.columns] = dummies
    task2_with_dummies.drop(col, axis=1, inplace=True)

# -1 потому что не считаем label
print('2.9:', len(task2_with_dummies.columns) - 1)

x_train, x_test, y_train, y_test = train_test_split(
    task2_with_dummies.drop('label', axis=1),
    task2_with_dummies['label'],
    test_size=0.2,
    random_state=15,
    stratify=task2_with_dummies['label']
)

scaler = MinMaxScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

knn3 = KNeighborsClassifier()
knn3.fit(x_train, y_train)
y_pred = knn3.predict(x_test)
print('2.10:', f1_score(y_test, y_pred))

for col in task2.columns:
    task2[col].replace('?', task2[col].mode().values[0], inplace=True)

for col in non_numeric_columns:
    dummies = pd.get_dummies(task2[col], drop_first=True)
    task2[dummies.columns] = dummies
    task2.drop(col, axis=1, inplace=True)

x_train, x_test, y_train, y_test = train_test_split(
    task2.drop('label', axis=1),
    task2['label'],
    test_size=0.2,
    random_state=15,
    stratify=task2['label']
)

scaler = MinMaxScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

knn4 = KNeighborsClassifier()
knn4.fit(x_train, y_train)
y_pred = knn4.predict(x_test)
print('2.11:', f1_score(y_test, y_pred))


#---------------------------------------------------

train = pd.read_csv('adult_data_train(2).csv')
test = pd.read_csv('adult_data_reserved.csv')


def clean_data(df):
    df.replace('?', np.nan, inplace=True)

    if 'label' in df.columns:
        for col in df.columns:
            df[col] = df.groupby("label")[col].transform(lambda x: x.fillna(x.mode()[0]))
    else:
        for col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return pd.get_dummies(
        df,
        columns=df.select_dtypes(include=[object]).columns,
        drop_first = True
    )


train = clean_data(train)
test = clean_data(test)
test = test.reindex(columns=train.columns, fill_value=0)

x_train = train.drop('label', axis=1)
y_train = train['label']
x_test = test.reindex(columns=x_train.columns, fill_value=0)

scaler = MinMaxScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

THRESHOLD = 0.05
corrs = [(y_train.corr(x_train[col]), col) for col in x_train.columns]
to_delete = [col for corr, col in corrs if corr < THRESHOLD]
x_train.drop(to_delete, axis=1, inplace=True)
x_test.drop(to_delete, axis=1, inplace=True)

knn = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 40)}
knn_gscv = GridSearchCV(knn, param_grid, cv=7)

print("Ебать столько времени предсказываться будет, чекай файл result и копируй оттуда ответ.\nИтоговые метрики: загрузка...")

knn_gscv.fit(x_train, y_train)
y_train_pred = knn_gscv.predict(x_train)
print('f1 score:', f1_score(y_train, y_train_pred))
y_pred = knn_gscv.predict(x_test)

with open('result', 'w') as file:
    file.write(str(list(y_pred)))


print(knn_gscv.best_score_)
print(knn_gscv.best_params_)