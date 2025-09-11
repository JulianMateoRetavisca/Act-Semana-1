from flask import Flask
from flask import render_template
from flask import request
import LRModel
from LRModel import CalculateGrade

app = Flask(__name__)

@app.route('/')
def index():
    Username = " Mateo"
    return render_template('index2.html', name=Username)

@app.route('/casoUno')
def casoUno():
    Username = " Mateo"
    return render_template('casoUno.html', name=Username)

@app.route('/casoDos')
def casoDos():
    Username = " Mateo"
    return render_template('casodos.html', name=Username)

@app.route('/casoTres')
def casoTres():
    Username = " Mateo"
    return render_template('casoTres.html', name=Username)

@app.route('/casoCuatro')
def casoCuatro():
    Username = " Mateo"
    return render_template('casoCuatro.html', name=Username)

@app.route('/LR', methods=["GET", "POST"])
def LR():
    calculateResult = None
    if request.method == "POST":
        hours = float(request.form["hours"])
        calculateResult = CalculateGrade(hours)
    return render_template("rl.html", result=calculateResult)

@app.route('/regresionConceptos')
def regresionConceptos():
    return render_template("rlconceptos.html")

if __name__ == "__main__":
    app.run(debug=True)
