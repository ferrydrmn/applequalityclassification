from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import SubmitField
from wtforms.validators import DataRequired

class ImageForm(FlaskForm):
    image = FileField('Unggah Citra Apel', validators=[DataRequired()])
    submit = SubmitField('Unggah')
    