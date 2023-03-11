from PIL import Image
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from IPython.display import Image



data = pd.read_csv('34_25.csv',header=None)


X_reduced = pd.read_csv('X_reduced_792.csv',header=None, sep=';').values.astype(float)
X_loadings = pd.read_csv('X_loadings_792.csv',header=None, sep=';').values.astype(float)



# выполнение метода главных компонент с двумя компонентами
pca = PCA(n_components=2)
pca.fit(data)

# нахождение новых координат для первого объекта
new_coordinates = pca.transform(data)
x1, y1 = new_coordinates[0]

# нахождение объясненной дисперсии для первых двух компонент
variance_ratio = sum(pca.explained_variance_ratio_[:2])

# нахождение минимального количества компонент, необходимых для объяснения 85% дисперсии
n_components = PCA(n_components=0.85).fit(data).n_components_

# выполнение кластеризации на основе двух компонент
kmeans = KMeans(n_clusters=5)
kmeans.fit(new_coordinates)


print("Задание 1\n")
# вывод результатов
print(f"Координата первого объекта по первой главной компоненте: {x1:.3f}")
print(f"Координата первого объекта по второй главной компоненте: {y1:.3f}")
print(
    f"Доля объясненной дисперсии при использовании первых двух главных компонент: {variance_ratio:.3f}")
print(
    f"Минимальное количество главных компонент для объяснения 85% дисперсии: {n_components}")
print(
    f"Количество групп объектов при использовании первых двух главных компонент: {len(set(kmeans.labels_))}")

print("\nЗадание 2\n")

# вычисление произведения матриц
X_restored = np.dot(X_reduced, X_loadings.T)

# приведение значений к диапазону от 0 до 255
X_restored = (X_restored - X_restored.min()) / \
    (X_restored.max() - X_restored.min()) * 255

# преобразование матрицы в изображение и сохранение его в файл
img = Image.fromarray(X_restored.astype('uint8'))
img.save('restored_image.png')
print("Сопоставьте фото с номером")


Image(filename='restored_image.png')