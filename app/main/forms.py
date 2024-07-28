from flask_wtf import FlaskForm
from wtforms.fields import IntegerField, SubmitField
from wtforms.validators import DataRequired, NumberRange

class TrainingForm(FlaskForm):
    """Form to select number of processors for training/testing."""
    num_processors = IntegerField('Number of Processors', validators=[DataRequired(), NumberRange(min=1, max=8)])
    submit = SubmitField('Start')
