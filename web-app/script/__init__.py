from flask import Flask

app = Flask(__name__)
app.config['SECRET_KEY'] = '5291628bb0b13cd0c676dfde280ba245'

from script import routes