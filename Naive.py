import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.sparse import hstack, csr_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_IMAGES_DIR = os.path.join(BASE_DIR, "static", "images")
os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)

def entrenar_y_graficar(csv_file=None):
    if csv_file is None:
        csv_file = os.path.join(BASE_DIR, "data", "naivebayes.csv")
    # asegúrate de que el archivo exista, si no, lanzar error claro
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV not found: {csv_file}")
    data = pd.read_csv(csv_file)

    X = data[['mensaje', 'prioridad', 'palabras_clave', 'hora']].copy()
    y = data['categoria']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    def concat_text(df):
        return (df['mensaje'].fillna('') + ' ' + df['palabras_clave'].fillna('')).values

    X_train_text = concat_text(X_train)
    X_test_text = concat_text(X_test)

    text_vec = TfidfVectorizer(max_features=2000)
    X_train_text_vec = text_vec.fit_transform(X_train_text)
    X_test_text_vec = text_vec.transform(X_test_text)

    cat_enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_cat = cat_enc.fit_transform(X_train[['prioridad']])
    X_test_cat = cat_enc.transform(X_test[['prioridad']])

    scaler = MinMaxScaler()
    X_train_num = scaler.fit_transform(X_train[['hora']])
    X_test_num = scaler.transform(X_test[['hora']])

    X_train_comb = hstack([X_train_text_vec, csr_matrix(X_train_cat), csr_matrix(X_train_num)])
    X_test_comb = hstack([X_test_text_vec, csr_matrix(X_test_cat), csr_matrix(X_test_num)])

    model = ComplementNB()
    model.fit(X_train_comb, y_train)
    y_pred = model.predict(X_test_comb)

    acc = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)

    img_name = f"naive_matrix_{int(time.time())}.png"
    img_path = os.path.join(STATIC_IMAGES_DIR, img_name)

    plt.figure(figsize=(6,5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel("Predicho, Categoria asignada")
    plt.ylabel("Real, Categorias Originales")
    plt.title("Matriz de Confusión - Naive Bayes")
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()

    # devolvemos ruta relativa dentro de static para que app.py la convierta en URL
    return {"accuracy": round(acc, 4), "image": img_path}

def predecir(csv_file, mensaje, prioridad, palabras_clave, hora, threshold=0.5):
    if not csv_file:
        csv_file = os.path.join(BASE_DIR, "data", "naivebayes.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV not found: {csv_file}")
    data = pd.read_csv(csv_file)

    X = data[['mensaje', 'prioridad', 'palabras_clave', 'hora']].copy()
    y = data['categoria']

    def concat_text(df):
        return (df['mensaje'].fillna('') + ' ' + df['palabras_clave'].fillna('')).values

    # entrenamos con todo el dataset
    text_vec = TfidfVectorizer(max_features=2000)
    X_text = concat_text(X)
    X_text_vec = text_vec.fit_transform(X_text)

    cat_enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat = cat_enc.fit_transform(X[['prioridad']])

    scaler = MinMaxScaler()
    X_num = scaler.fit_transform(X[['hora']])

    X_comb = hstack([X_text_vec, csr_matrix(X_cat), csr_matrix(X_num)])
    model = ComplementNB()
    model.fit(X_comb, y)

    # procesar entrada nueva
    input_df = pd.DataFrame([{
        "mensaje": mensaje,
        "prioridad": prioridad,
        "palabras_clave": palabras_clave,
        "hora": hora
    }])
    input_text = concat_text(input_df)
    input_text_vec = text_vec.transform(input_text)
    input_cat = cat_enc.transform([[prioridad]])
    input_num = scaler.transform([[hora]])
    input_comb = hstack([input_text_vec, csr_matrix(input_cat), csr_matrix(input_num)])

    probs = model.predict_proba(input_comb)[0]
    clases = model.classes_

    # si las clases contienen 'Sí' usamos su posición, si no suponemos clase positiva en índice 1
    if "Sí" in clases:
        idx_si = list(clases).index("Sí")
        prob_si = float(probs[idx_si])
        pred = "Sí" if prob_si >= threshold else "No"
    else:
        # fallback: tomar la probabilidad de la clase predicha
        pred_idx = int(model.predict(input_comb)[0] in clases)  # not used directly
        prob_si = float(max(probs))
        pred = model.predict(input_comb)[0]

    interpretacion = f"Con threshold={threshold}, la predicción ajusta su sensibilidad al umbral elegido."

    return {
        "prediccion": pred,
        "probabilidad": round(prob_si, 4),
        "threshold": threshold,
        "interpretacion": interpretacion
    }
