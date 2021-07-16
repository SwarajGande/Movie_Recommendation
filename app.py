from flask import Flask, render_template, request
import numpy as np
import pickle
import task17
app = Flask(__name__)

model = pickle.load(open("recomm.pkl","rb"))
@app.route("/")
def home():
  return render_template("index.html")

@app.route("/recommend", methods = ["POST"])
def recommend():
  movie_title =  request.form["movie"]
  pred = task17.recommendation(movie_name = movie_title)
  return render_template("index.html",data = pred, len=len(pred))

if __name__ == "__main__":
    app.run(debug = True)