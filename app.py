from flask import Flask, render_template, request, redirect, url_for, flash
import os
from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import LRModel
from LRModel import CalculateGrade
import LogRep
import Naivebayes

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

###########################################
# CONCEPTOS
###########################################
@app.route('/regresionConceptos')
def regresionConceptos():
    return render_template("rlconceptos.html")


@app.route('/LogisConceptos')
def LogisConceptos():
    return render_template("logisconceptos.html")

###########################################
# REGRESIÓN LINEAL
###########################################
@app.route('/LR', methods=["GET", "POST"])
def LR():
    # Cargamos siempre datos actualizados
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

    # Generamos carpeta para guardar gráfico
    image_path = os.path.join('static', 'images')
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    # Graficar datos y línea de regresión
    plt.figure(figsize=(8, 6))
    plt.scatter(df["Distancia Recorrida"], df["Precio"], color='blue', s=80, label='Datos')
    x_vals = np.linspace(df["Distancia Recorrida"].min(), df["Distancia Recorrida"].max(), 100)
    y_vals = [CalculateGrade(x, 1) for x in x_vals]  
    plt.plot(x_vals, y_vals, color='orange', linewidth=2, label='Línea de Regresión')

    plt.title('Gráfico de Dispersión y Línea de Regresión')
    plt.xlabel('Distancia Recorrida')
    plt.ylabel('Precio')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    graph_file = os.path.join(image_path, 'grafico.png')
    plt.savefig(graph_file)
    plt.close()

    return render_template(
        "rl.html",
        result=calculateResult,
        graph_url=os.path.join('static', 'images', 'grafico.png'),
        media=media,
        mediana=mediana,
        message=message  
    )

###########################################
# REGRESIÓN LOGÍSTICA
###########################################
@app.route('/Lc', methods=["GET", "POST"])
def Lc():
    result = None
    prob = None

    # Entrenar modelo (siempre en cada request)
    model, scaler, x_test, y_test = LogRep.entrenar_modelo()

    # Evaluar modelo y generar matriz de confusión ( se genera en GET y POST)
    metrics = LogRep.evaluar_modelo(model, x_test, y_test, filename="static/images/confusion_matrix.png")

    if request.method == "POST":
        # Recibir datos del formulario
        horas = float(request.form["horas"])
        calorias = float(request.form["calorias"])
        sexo = int(request.form["sexo"])
        pantalla = float(request.form["pantalla"])

        features = [horas, calorias, sexo, pantalla]
        result, prob = LogRep.Predecir(model, scaler, features)

        #  Volvemos a evaluar después de predecir para regenerar matriz actualizada
        metrics = LogRep.evaluar_modelo(model, x_test, y_test, filename="static/images/confusion_matrix.png")

    return render_template(
        "Lc.html",
        metrics=metrics,
        result=result,
        prob=prob,
        confusion_matrix=url_for('static', filename='images/confusion_matrix.png')
    )

###########################################
# NAIVE BAYES 
###########################################
@app.route("/naivebayes", methods=["GET", "POST"])
def naive_bayes():
    resultados = Naivebayes.entrenar_y_graficar("naivebayes.csv")
    prediccion = probabilidad = interpretacion = None

    if request.method == "POST":
        mensaje = request.form["mensaje"]
        prioridad = request.form["prioridad"]
        palabras_clave = request.form["palabras_clave"]
        hora = float(request.form["hora"])
        threshold = float(request.form.get("threshold", 0.5))
        pred = Naivebayes.predecir("naivebayes.csv", mensaje, prioridad, palabras_clave, hora, threshold)
        prediccion = pred["prediccion"]
        probabilidad = pred["probabilidad"]
        interpretacion = pred["interpretacion"]

    return render_template("NaiveBayes.html",
                           accuracy=resultados["accuracy"],
                           image=resultados["image"],
                           prediccion=prediccion,
                           probabilidad=probabilidad,
                           interpretacion=interpretacion)

@app.route("/Practicos")
def Practicos():
    return render_template("AlgoritClas.html")

if __name__ == '__main__':
    app.run(debug=True)

