from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
import numpy as np
from sklearn.metrics import mean_absolute_error


def run_model_light(X, y, cv, model, scoring="accuracy"):
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    print(np.mean(scores))


def run_model(X, y, cv, model, cv_type):
    scores = []

    if cv_type == 'KFold':
        cv_split = cv.split(y)

    elif cv_type == 'StratifiedKFold':
        cv_split = cv.split(X, y)

    else:
        return 'Error'

    for train_idx, test_idx in cv_split:
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])

        print('Amount of labels:', np.bincount(y[train_idx]), 'train:', y[train_idx].size, ' test:', y[test_idx].size)

        score = mean_absolute_error(y[test_idx], y_pred)
        scores.append(score)

    return np.mean(scores)


def lesson_3():
    X, y = load_wine(return_X_y=True)

    print('\n KFold \n')

    cv = KFold(n_splits=2, random_state=2022, shuffle=True)

    model = KNeighborsClassifier()

    run_model(X=X, y=y, cv=cv, model=model, cv_type='KFold')

    print('\n StratifiedKFold  \n')

    cv = StratifiedKFold(n_splits=2, random_state=2022, shuffle=True)

    model = KNeighborsClassifier()

    run_model(X=X, y=y, cv=cv, model=model, cv_type='StratifiedKFold')

    model = KNeighborsClassifier()

    for i in [3, 5, 10]:
        print('CV:', i)
        run_model_light(X=X, y=y, model=model, cv=i)
        print('\n\n')


if __name__ == '__main__':
    lesson_3()

