import os
import xlrd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input


def extract_data(type):
    book = xlrd.open_workbook('data.xlsx')
    sheet = book.sheet_by_name('Sheet1')
    data = [[sheet.cell_value(r, c) for c in range(sheet.ncols)] for r in range(sheet.nrows)]
    data = data[1:]
    x = []
    y = []
    for entry in data:
        x.append(entry[2:type + 1])
        y.append(int(entry[type + 1]))
    y = pd.get_dummies(y)
    return np.array(x), np.array(y)


def create_model(type):
    model = Sequential()
    model.add(Dense(5, input_dim=type - 1, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def predict_step_2():
    x, y = extract_data(type=2)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    model = create_model(type=2)
    model.fit(x_train, y_train, epochs=200, batch_size=5)
    _, accuracy = model.evaluate(x_test, y_test)
    return accuracy * 100


def predict_step_3():
    x, y = extract_data(type=3)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    model = create_model(type=3)
    model.fit(x_train, y_train, epochs=200, batch_size=5)
    _, accuracy = model.evaluate(x_test, y_test)
    return accuracy * 100


def predict_step_4():
    x, y = extract_data(type=4)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    model = create_model(type=4)
    model.fit(x_train, y_train, epochs=200, batch_size=5)
    _, accuracy = model.evaluate(x_test, y_test)
    return accuracy * 100


def predict_step_5():
    x, y = extract_data(type=5)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    model = create_model(type=5)
    model.fit(x_train, y_train, epochs=200, batch_size=5)
    _, accuracy = model.evaluate(x_test, y_test)
    return accuracy * 100


def predict_step_6():
    x, y = extract_data(type=6)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    model = create_model(type=6)
    model.fit(x_train, y_train, epochs=200, batch_size=5)
    _, accuracy = model.evaluate(x_test, y_test)
    return accuracy * 100


if __name__ == "__main__":
    accuracy_step_2 = round(predict_step_2(), 2)
    accuracy_step_3 = round(predict_step_3(), 2)
    accuracy_step_4 = round(predict_step_4(), 2)
    accuracy_step_5 = round(predict_step_5(), 2)
    accuracy_step_6 = round(predict_step_6(), 2)
    print(f"Prediction of choice #2 accuracy: {accuracy_step_2}%")
    print(f"Prediction of choice #3 accuracy: {accuracy_step_3}%")
    print(f"Prediction of choice #4 accuracy: {accuracy_step_4}%")
    print(f"Prediction of choice #5 accuracy: {accuracy_step_5}%")
    print(f"Prediction of choice #6 accuracy: {accuracy_step_6}%")

