from flask import Blueprint,Flask, flash, redirect, render_template, request, url_for, jsonify
import json
from webapp.helpers import load_bert_model, error_view

main_controller = Blueprint('main', __name__, template_folder="../views", static_folder = '../static')

model = load_bert_model()


@main_controller.route("/")
def index():
    return render_template("index.html")


@main_controller.route("/api/bert", methods=["POST"])
def bert_api():
    """Return a json list of predictions for the given sentence query"""

    # if user reached route via POST (as by submitting a form via POST)
    if request.method == "POST":
        if not request.form['text']:
            return error_view("Must enter text to get predictions.")
        try:
            text = request.form['text']
            result = model.predict(text)
            val = json.dumps(result)
            return jsonify({
                'text': text,
                'result': val
            })
        except:
            import traceback
            print(traceback.format_exc())
            return jsonify({
                'error': "Something went wrong...try again"
            })