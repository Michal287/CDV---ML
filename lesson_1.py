from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import matplotlib.pyplot as plt


def score_knn(X_train, y_train, X_test, y_test, k=3, title=False):
    n_neighbors = np.arange(1, k + 1)

    train_accuracy = np.empty(len(n_neighbors))
    test_accuracy = np.empty(len(n_neighbors))

    for i, k in enumerate(n_neighbors):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        train_accuracy[i] = knn.score(X_train, y_train)
        test_accuracy[i] = knn.score(X_test, y_test)

    if title:
        plt.title(title)

    plt.plot(n_neighbors, test_accuracy, label='Testing')
    plt.plot(n_neighbors, train_accuracy, label='Training')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.show()

    max_result_idx = np.argmax(test_accuracy, axis=0)
    print('Best result for n_neighbors =', max_result_idx + 1)


def lesson_1():

    X, y = load_wine(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=2022)

    model = DummyClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("DummyClassifier:", accuracy_score(y_test, y_pred))

    joblib.dump(model, 'models/lesson_1.pkl')

    loaded_model = joblib.load(open('models/lesson_1.pkl', 'rb'))

    y_pred = loaded_model.predict(X_test)
    accuracy_score(y_test, y_pred)

    score_knn(X_train, y_train, X_test, y_test, k=10, title='Without StandardScaler')

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=2022)

    score_knn(X_train, y_train, X_test, y_test, k=10, title='StandardScaler')


if __name__ == '__main__':
    lesson_1()