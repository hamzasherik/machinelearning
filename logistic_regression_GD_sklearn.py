from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)

standard_scalar = StandardScaler()
standard_scalar.fit(X_train)
X_train = standard_scalar.transform(X_train)
X_test = standard_scalar.transform(X_test)

logistic_regression = LogisticRegression(multi_class='ovr')
logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)

print(logistic_regression.score(X_test, y_pred))