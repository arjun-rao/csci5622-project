
from flask import Flask, flash, redirect, render_template, request, session, url_for


# setup models and controllers
from controllers.main import main_controller

# configure application to use views folder for views
app = Flask(__name__, template_folder='views')

# ensure responses aren't cached
if not app.config["DEBUG"]:
    @app.after_request
    def after_request(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Expires"] = 0
        response.headers["Pragma"] = "no-cache"
        return response




# register controllers
app.register_blueprint(main_controller)




