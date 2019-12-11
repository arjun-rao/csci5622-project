import os
import sys
sys.path.append(os.getcwd() + '/..')
from bert.bert import BertPredictor
from flask import Blueprint,Flask, flash, redirect, render_template, request, url_for, jsonify

from webapp.settings import BERT_MODEL_PATH

def error_view(top="", bottom=""):
    """Renders message as an apology to user."""
    def escape(s):
        """
        Escape special characters.

        https://github.com/jacebrowning/memegen#special-characters
        """
        for old, new in [("-", "--"), (" ", "-"), ("_", "__"), ("?", "~q"),
            ("%", "~p"), ("#", "~h"), ("/", "~s"), ("\"", "''")]:
            s = s.replace(old, new)
        return s
    return render_template("apology.html", top=escape(top), bottom=escape(bottom))

def load_bert_model():
    """loads and instantiates bert model for flask server"""
    model = BertPredictor()
    model.load_model(BERT_MODEL_PATH)
    return model
