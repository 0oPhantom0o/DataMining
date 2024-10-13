import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imodels import RuleFitClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_dataset(file_path):
    DataFrame = pd.read_csv(file_path)
    return DataFrame


def split_dataset(DataFrame, features, target, test_size, random_state):
    X = DataFrame[features]
    y = DataFrame[target]
    input, sample_test, output, real_data = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return input, sample_test, output, real_data


def naivebar(input, sample_test, output, real_data):
    classifier = GaussianNB()


    classifier.fit(input, output)

    predictions = classifier.predict(sample_test)

    accuracy = accuracy_score(real_data, predictions)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(real_data, predictions))
    data = "Naive Bayes Accuracy: {}\n".format(accuracy * 100)
    # Insert data into file

    insert_data_into_file(data)


def decision_tree(input, sample_test, output, real_data,random_state):
    param_grid = {
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    classifier = DecisionTreeClassifier(random_state=random_state)

    grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(input, output)

    best_clf = grid_search.best_estimator_

    y_pred = best_clf.predict(sample_test)

    accuracy = accuracy_score(real_data, y_pred)

    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(real_data, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(real_data, y_pred))
    data = "Decision Tree Accuracy: {}\n".format(accuracy * 100)

    insert_data_into_file(data)


def svm(input, sample_test, output, real_data, random_state):
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input)
    sample_test_scaled = scaler.transform(sample_test)

    logistic_regression = LogisticRegression(random_state=random_state)
    logistic_regression.fit(input_scaled, output)
    y_pred_lr = logistic_regression.predict(sample_test_scaled)

    svm_classifier = SVC(kernel='linear', random_state=random_state)
    svm_classifier.fit(input_scaled, output)

    accuracy = accuracy_score(real_data, y_pred_lr)
    print("Logistic Regression Accuracy:", accuracy)
    print("\nLogistic Regression Classification Report:")
    print(classification_report(real_data, y_pred_lr))
    print("\nLogistic Regression Confusion Matrix:")
    print(confusion_matrix(real_data, y_pred_lr))
    data = "SVM Accuracy: {}\n".format(accuracy * 100)
    insert_data_into_file(data)


def rulinduction(input, sample_test, output, real_data, random_state):
    rf_model = RuleFitClassifier(random_state=random_state)

    rf_model.fit(input, output)

    y_pred = rf_model.predict(sample_test)

    accuracy = accuracy_score(real_data, y_pred)
    print("Accuracy:", accuracy * 100)

    print("\nClassification Report:")
    print(classification_report(real_data, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(real_data, y_pred))

    rules = rf_model._get_rules()
    print("\nLearned Rules:")
    for rule in rules:
        print(rule)
    data = "Rule Induction Accuracy: {}\n".format(accuracy * 100)
    insert_data_into_file(data)


def insert_data_into_file(data):
    file_path = 'output.txt'

    with open(file_path, 'a') as file:
        file.write(data)
        file.write('\n')


file_path = 'output.txt'
DataFrame = load_dataset("heart.csv")
random_state = 42
input, sample_test, output, real_data = split_dataset(DataFrame,
                                                      features=['Age', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
                                                           'RestingBP'], target='HeartDisease', test_size=0.3,
                                                      random_state=42)
#naivebar(input, sample_test, output, real_data)
# decision_tree(input, sample_test, output, real_data, random_state)
svm(input, sample_test, output, real_data, random_state)
# rulinduction(input, sample_test, output, real_data, random_state)
