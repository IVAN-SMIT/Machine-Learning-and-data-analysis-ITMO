import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


df = pd.read_csv('pulsar_stars_new.csv')

dfq = df.query('(TG == 0 & MIP >= 68.1640625  & MIP <= 172.5625) | (TG == 1 & MIP >= 6.4140625 & MIP <= 113.6484375)')
print(dfq.STDIP.mean())

dfs = dfq.sort_values(by='SIP')
X = dfs.drop('TG',axis=1)
Y = dfs['TG']

x,xt,y,yt = train_test_split(X,Y,random_state=23,test_size=0.2,stratify=Y)

print(x.SIP.mean())

sc = MinMaxScaler().fit(x)
x[x.columns] = sc.transform(x)
xt[xt.columns] = sc.transform(xt)

print(x.STDIP.mean())
print()

pred = LogisticRegression(random_state = 23).fit(x,y).predict(xt)

print(confusion_matrix(yt,pred)[1][1])
print(confusion_matrix(yt,pred)[1][0])
print(f1_score(yt,pred))
print()

pred = KNeighborsClassifier(n_neighbors=3).fit(x,y).predict(xt)

print(confusion_matrix(yt,pred)[1][1])
print(confusion_matrix(yt,pred)[0][1])
print(f1_score(yt,pred))
