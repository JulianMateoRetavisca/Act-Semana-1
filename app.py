from flask import Flask, render_template, request, redirect, url_for, flash
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import LRModel
from LRModel import CalculateGrade
import LogRep
import Naive

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_IMAGES_DIR = os.path.join(BASE_DIR, "static", "images")

app = Flask(__name__)

@app.route('/')
def index():
    Username = "Mateo"
    return render_template('index2.html', name=Username)

@app.route('/casoUno')
def casoUno():
    Username = "Mateo"
    return render_template('casoUno.html', name=Username)

@app.route('/casoDos')
def casoDos():
    Username = "Mateo"
    return render_template('casodos.html', name=Username)

@app.route('/casoTres')
def casoTres():
    Username = "Mateo"
    return render_template('casoTres.html', name=Username)

@app.route('/casoCuatro')
def casoCuatro():
    Username = "Mateo"
    return render_template('casoCuatro.html', name=Username)

@app.route('/regresionConceptos')
def regresionConceptos():
    return render_template("rlconceptos.html")

@app.route('/LogisConceptos')
def LogisConceptos():
    return render_template("logisconceptos.html")

@app.route('/LR', methods=["GET", "POST"])
def LR():
    df = LRModel.df
    media = df["Precio"].mean()
    mediana = df["Precio"].median()
    calculateResult = None
    distancia = None
    pasajeros = None
    message = None
    if request.method == "POST":
        if "precio" in request.form:
            distancia = float(request.form["distancia"])
            pasajeros = float(request.form["pasajeros"])
            precio = float(request.form["precio"])
            message = LRModel.UpdateData(distancia, pasajeros, precio)
        else:
            distancia = float(request.form["distancia"])
            pasajeros = float(request.form["pasajeros"])
            calculateResult = CalculateGrade(distancia, pasajeros)
    os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(df["Distancia Recorrida"], df["Precio"], s=80, label='Datos')
    x_vals = np.linspace(df["Distancia Recorrida"].min(), df["Distancia Recorrida"].max(), 100)
    y_vals = [CalculateGrade(x, 1) for x in x_vals]
    plt.plot(x_vals, y_vals, linewidth=2, label='Línea de Regresión')
    plt.title('Gráfico de Dispersión y Línea de Regresión')
    plt.xlabel('Distancia Recorrida')
    plt.ylabel('Precio')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    graph_file = os.path.join(STATIC_IMAGES_DIR, 'grafico.png')
    plt.savefig(graph_file)
    plt.close()
    return render_template(
        "rl.html",
        result=calculateResult,
        graph_url=url_for('static', filename='images/grafico.png'),
        media=media,
        mediana=mediana,
        message=message
    )

@app.route('/Lc', methods=["GET", "POST"])
def Lc():
    result = None
    prob = None
    os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)
    model, scaler, x_test, y_test = LogRep.entrenar_modelo()
    confusion_path = os.path.join(STATIC_IMAGES_DIR, "confusion_matrix.png")
    metrics = LogRep.evaluar_modelo(model, x_test, y_test, filename=confusion_path)
    if request.method == "POST":
        horas = float(request.form["horas"])
        calorias = float(request.form["calorias"])
        sexo = int(request.form["sexo"])
        pantalla = float(request.form["pantalla"])
        features = [horas, calorias, sexo, pantalla]
        result, prob = LogRep.Predecir(model, scaler, features)
        metrics = LogRep.evaluar_modelo(model, x_test, y_test, filename=confusion_path)
    return render_template(
        "Lc.html",
        metrics=metrics,
        result=result,
        prob=prob,
        confusion_matrix=url_for('static', filename='images/confusion_matrix.png')
    )

@app.route("/naivebayes", methods=["GET", "POST"])
def naive_bayes():
    os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)
    csv_path = os.path.join(BASE_DIR, "naivebayes.csv")
    resultados = Naive.entrenar_y_graficar(csv_path)
    prediccion = probabilidad = interpretacion = None
    if request.method == "POST":
        mensaje = request.form["mensaje"]
        prioridad = request.form["prioridad"]
        palabras_clave = request.form["palabras_clave"]
        hora = float(request.form["hora"])
        threshold = float(request.form.get("threshold", 0.5))
        pred = Naive.predecir(csv_path, mensaje, prioridad, palabras_clave, hora, threshold)
        prediccion = pred.get("prediccion")
        probabilidad = pred.get("probabilidad")
        interpretacion = pred.get("interpretacion")
    image_url = None
    image_path = resultados.get("image") if isinstance(resultados, dict) else None
    if image_path:
        image_url = url_for('static', filename='images/' + os.path.basename(image_path))
    return render_template("NaiveBayes.html",
                           accuracy=resultados.get("accuracy") if isinstance(resultados, dict) else None,
                           image=image_url,
                           prediccion=prediccion,
                           probabilidad=probabilidad,
                           interpretacion=interpretacion)

@app.route("/Practicos")
def Practicos():
    return render_template("AlgoritClas.html")

if __name__ == '__main__':
    app.run(debug=True)
