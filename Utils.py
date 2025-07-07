import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, BatchNormalization

_df = pd.read_csv('creditcard.csv')

FEATURES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
TARGET = 'Class'

_rs_amount = RobustScaler().fit(_df[['Amount']])
_mm_time = MinMaxScaler().fit(_df[['Time']])


def _make_xy(df: pd.DataFrame):

    t = _mm_time.transform(df[['Time']])
    a = _rs_amount.transform(df[['Amount']])
    vs = df[[f'V{i}' for i in range(1, 29)]].values
    X = np.hstack([t, vs, a])
    y = df[TARGET].values if TARGET in df.columns else None
    return X, y


def get_splits(sampling: str = 'imbalanced'):

    df0 = _df[_df[TARGET] == 0]
    df1 = _df[_df[TARGET] == 1]
    if sampling == 'balanced':
        df0 = df0.sample(n=len(df1), random_state=1)
    df_all = pd.concat([df0, df1]).sample(frac=1, random_state=1)

    X, y = _make_xy(df_all)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, random_state=1, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=1, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_and_evaluate_all(sampling: str = 'imbalanced'):

    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(sampling)
    results = {}

    # Logistic Regression
    lr = LogisticRegression(random_state=0).fit(X_train, y_train)
    results['LR'] = _eval_model(lr, X_test, y_test)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    rf.fit(X_train, y_train)
    results['RF'] = _eval_model(rf, X_test, y_test)

    # Gradient Boosting
    gbc = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=0)
    gbc.fit(X_train, y_train)
    results['GBC'] = _eval_model(gbc, X_test, y_test)

    # Shallow Neural Network
    nn = Sequential([
        InputLayer((X_train.shape[1],)),
        Dense(2, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    nn.compile(optimizer='adam', loss='binary_crossentropy')
    nn.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, verbose=0)
    results['NN'] = _eval_keras(nn, X_test, y_test)

    return results


def predict_single(sampling: str = 'imbalanced', input_array: np.ndarray = None, **input_features):

    if input_array is not None:
        arr = np.array(input_array).flatten()
        if arr.shape[0] != len(FEATURES):
            raise ValueError(f"input_array must have {len(FEATURES)} values, got {arr.shape[0]}")
        input_features = dict(zip(FEATURES, arr))
    else:
        missing = set(FEATURES) - set(input_features.keys())
        if missing:
            raise ValueError(f"Missing features: {missing}")

    results = train_and_evaluate_all(sampling)
    best_name = max(results, key=lambda m: results[m]['f1'])

    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(sampling)
    X_full = np.vstack([X_train, X_val])
    y_full = np.hstack([y_train, y_val])

    if best_name == 'LR':
        model = LogisticRegression(random_state=0).fit(X_full, y_full)
    elif best_name == 'RF':
        model = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0).fit(X_full, y_full)
    elif best_name == 'GBC':
        model = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=0).fit(X_full, y_full)
    else:
        model = Sequential([
            InputLayer((X_full.shape[1],)),
            Dense(2, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit(X_full, y_full, epochs=5, verbose=0)

    df_input = pd.DataFrame([input_features])
    X_input, _ = _make_xy(df_input)

    # Predict
    if hasattr(model, 'predict_proba'):
        pred = int(model.predict(X_input)[0])
    else:
        pred = int(model.predict(X_input).flatten()[0] > 0.5)

    return best_name, pred


def _eval_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'cm': confusion_matrix(y_test, y_pred)
    }


def _eval_keras(model, X_test, y_test):
    y_proba = model.predict(X_test).flatten()
    y_pred = (y_proba > 0.5).astype(int)
    return {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'cm': confusion_matrix(y_test, y_pred)
    }