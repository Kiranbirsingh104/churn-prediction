import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train_model():
    data = pd.read_csv("data.csv")

    # Features & target
    X = data.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
    y = data['Exited'].values
    # Encode Gender
    le = LabelEncoder()
    X['Gender'] = le.fit_transform(X['Gender'])

    # One-hot encode Geography
    ct = ColumnTransformer(
        transformers=[('geo', OneHotEncoder(drop='first'), ['Geography'])],
        remainder='passthrough'
    )
    X = ct.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # ANN model
    model = Sequential()
    model.add(Dense(units=6, activation='sigmoid', input_dim=X_train.shape[1]))
    model.add(Dense(units=6, activation='sigmoid'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=25, epochs=10)

    print("Accuracy:", model.evaluate(X_test, y_test)[1])

    return model, sc, ct
