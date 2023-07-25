from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

perceptron = Perceptron(eta0=0.01, random_state=1)
perceptron.fit(X_train_std, y_train)
y_pred = perceptron.predict(X_test_std)
print(perceptron.score(X_train_std, y_train))