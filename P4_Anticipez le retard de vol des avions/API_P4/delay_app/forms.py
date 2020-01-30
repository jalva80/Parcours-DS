from flask_wtf import FlaskForm
# from wtforms.fields.html5 import StringField, DateField, DateTimeField, SelectField, SubmitField
from wtforms import StringField, DateField, DateTimeField, SelectField, SubmitField
from wtforms.validators import DataRequired
from . import utils
from flask import flash

# from wtforms import Form, DateField
from wtforms_components import TimeField


class TripForm(FlaskForm):
    cies, airports = utils.init_sel()

    date_vol = DateField('Date du vol', validators=[DataRequired()])
    heure_vol = TimeField('Heure du vol', validators=[DataRequired()])
    cie_id = SelectField('Compagnie aérienne', choices=cies, validators=[DataRequired()])
    dep_id = SelectField('Aéroport de départ', choices=airports, validators=[DataRequired()])
    arr_id = SelectField('Aéroport d\'arrivée', choices=airports, validators=[DataRequired()])
    submit = SubmitField('Prédire le retard')
