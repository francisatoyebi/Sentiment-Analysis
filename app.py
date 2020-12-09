from flask import Flask
from flask import request,render_template,redirect,url_for,jsonify

from Classifier import sentiment_classifier

app = Flask(__name__,template_folder="templates")

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict",methods=["POST"])
def predict():
    #checks if method is post
    if request.method == "POST":  
        word = request.form["sentence"]
        result = sentiment_classifier(word)
        return render_template("index.html",result=result)
        #LOGIC GOES HERE

@app.route("/results",methods=["POST"])
def results():
    word = request.get_json(force=True)
    result = sentiment_classifier(word.values())
    return jsonify(result)
