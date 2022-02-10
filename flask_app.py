import scripts
from flask import Flask
from flask_cors import CORS
from flask import render_template, request


app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET", "POST"])
def default():
    if request.method == "POST":
        sentence = request.form["sentence"]
        prediction = scripts.predict(sentence)
        return render_template(
            "index.html",
            previous_sentence=sentence,
            prediction=prediction)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run()
