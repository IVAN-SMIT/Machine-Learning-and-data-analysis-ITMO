from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd


def mean(arr):
    return sum(arr)/len(arr)

def lineur_regression(x_arr, y_arr):
    if (len(x_arr) != len(y_arr)):
        raise Exception("ERROR: length not equals")
    x_mean = mean(x_arr)
    y_mean = mean(y_arr)
    num_sum, den_sum = 0, 0
    for i in range(len(x_arr)):
        num_sum += (x_arr[i] - x_mean) * (y_arr[i] - y_mean)
        den_sum += (x_arr[i] - x_mean) ** 2
    theta_1 = num_sum / den_sum
    theta_0 = y_mean - theta_1 * x_mean

    # caclulate R
    num_sum, den_sum = 0, 0
    for i in range(len(x_arr)):
        num_sum += (y_arr[i] - theta_0 - theta_1 * x_arr[i]) ** 2
        den_sum += (y_arr[i] - y_mean) ** 2
    r_pow_2 = 1 - num_sum / den_sum

    print("theta_1 = ", theta_1)
    print("theta_0 = ", theta_0)
    print("r_pow_2 = ", r_pow_2)

# task-1
x_arr = [6, 14, 10, 16, 2, 1, 21, 7, 25, 12]
y_arr = [12, 23, 32, 44, 5, 5, 40, 15, 40, 34]
x_mean = mean(x_arr)
y_mean = mean(y_arr)
print("Выборочное среднее X: ", x_mean)
print("Выборочное среднее Y: ", y_mean)
lineur_regression(x_arr, y_arr)
#------------------------------------------------------------------------------
# task-2

# Параметры
df = pd.read_csv('fish_train.csv')
test_size = 0.2
random_state=15

def get_splits():
    species = df['Species']
    x = df.drop(['Weight', 'Species'], axis=1)
    y = df['Weight']

    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=species)

# Построение базовой модели

x_train, x_test, y_train, y_test = get_splits()

print('Среднее колонки Width:', x_train['Width'].mean())

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print('R2 оценка для тестового набора:', r2_score(y_test, y_pred))

# Добавление предварительной обратботки признаков

# Скорее всего, тройка будет Length1...3, ну все равно будьте внимательны
print('Матрица корреляции:')

x_train, x_test, y_train, y_test = get_splits()

print(df.drop('Species', axis=1).corr())
pca = PCA(n_components=1, svd_solver='full')
pca.fit(x_train[['Length1', 'Length2', 'Length3']])
print('Доля объясненной дисперсии:', pca.explained_variance_ratio_)
df['Lengths'] = pca.transform(df[['Length1', 'Length2', 'Length3']])
df.drop(['Length1', 'Length2', 'Length3'], axis=1, inplace=True)

x_train, x_test, y_train, y_test = get_splits()

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print('R2 оценка для тестового набора:', r2_score(y_test, y_pred))

# Модификация признаков

df[['Width', 'Height', 'Lengths']] = df[['Width', 'Height', 'Lengths']].apply(lambda x: x**3)

x_train, x_test, y_train, y_test = get_splits()

print('Среднее Width после возведения в куб:', x_train['Width'].mean())

unique_species = df['Species'].unique()
for i in unique_species:
    plt.scatter(df[df['Species'] == i]['Width'], df[df['Species'] == i]['Weight'], label=i)
plt.legend()
plt.xlabel('Width')
plt.ylabel('Weight')
plt.show()

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print('R2 оценка для тестового набора:', r2_score(y_test, y_pred))

dummies = pd.get_dummies(df['Species'])
df[list(dummies.columns)] = dummies
x_train, x_test, y_train, y_test = get_splits()
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print('R2 оценка для тестового набора:', r2_score(y_test, y_pred))
df.drop(list(dummies.columns), axis=1, inplace=True)

dummies = pd.get_dummies(df['Species'], drop_first=True)
df[list(dummies.columns)] = dummies
x_train, x_test, y_train, y_test = get_splits()
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print('R2 оценка для тестового набора:', r2_score(y_test, y_pred))





#--------------------------------------------------------------


#3 task


df_train = pd.read_csv('fish_train.csv')
df_test = pd.read_csv('fish_reserved.csv')
#pca = PCA(n_components=1, svd_solver='full')
#pca.fit(df_train[['Length1', 'Length2', 'Length3']])

def clean(df):
    #df['Lengths'] = df[['Length1', 'Length2', 'Length3']] @ pca.components_[0].T
    #df.drop(['Length1', 'Length2', 'Length3'], axis=1, inplace=True)

    #df[['Width', 'Height', 'Lengths']] = df[['Width', 'Height', 'Lengths']].apply(lambda x: x**3)

    df[['Width', 'Height', 'Length1', 'Length2', 'Length3']] = df[['Width', 'Height', 'Length1', 'Length2', 'Length3']].apply(lambda x: x**2)

    dummies = pd.get_dummies(df['Species'], drop_first=True)
    df[list(dummies.columns)] = dummies
    df.drop(['Species'], axis=1, inplace=True)

clean(df_train)
clean(df_test)

x_train = df_train.drop(['Weight'], axis=1)
y_train = df_train['Weight']
x_test = df_test

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
print(list(y_pred))