from flask_wtf import FlaskForm
# from wtforms.fields.html5 import StringField, DateField, DateTimeField, SelectField, SubmitField
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from wtforms.widgets import TextArea
from . import utils
from flask import flash


class NewMsgForm(FlaskForm):
    title_raw = StringField(u'Titre', widget=TextArea(), validators=[DataRequired()])
    body_raw = StringField(u'Message', widget=TextArea(), validators=[DataRequired()])

    submit = SubmitField('Proposer des catégories!')
