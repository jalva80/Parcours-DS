# from flask import Flask
from . import utils
from flask import render_template, flash, redirect, url_for
from delay_app import app
from delay_app.forms import TripForm

# app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = TripForm()

    if form.validate_on_submit():
        pred_delay = utils.delay_pred(form)
        if pred_delay:
            flash('La prédiction du retard du vol est de {} minute(s)'.format(pred_delay))
        else:
            flash('Une erreur s\'est produite lors de l\'application de la regression')

    return render_template('predict.html', title='Prédiction de retard', form=form)

if __name__ == "__main__":
    app.run()
