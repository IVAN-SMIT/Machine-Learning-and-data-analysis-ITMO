import pandas as pd

# загрузка данных из файла csv
df = pd.read_csv(r'ROSSTAT_SALARY_RU.csv', encoding="UTF-8")

# удаление строк с регионами, которые не нужны
df = df.set_index("region_name").drop(['Республика Северная Осетия-Алания','Томская область','Ивановская область'], axis = 0)

# сортировка данных по возрастанию и построение вариационного ряда
var_series = df['salary'].sort_values().reset_index(drop=True)


# вывод ответов, где X[n] = var_series[n-1]

print(f"X[12] = {var_series[11]}; X[19] = {var_series[18]}; X[59] = {var_series[58]}")
print(f"Выборочное среднее: {round(var_series.mean(), 3)}")
print(f"Медиана: {var_series.median() }")






