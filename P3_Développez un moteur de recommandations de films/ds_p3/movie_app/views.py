from flask import Flask
from . import utils

app = Flask(__name__)

# @app.route('/')
# def index():
#     return "Salut la compagnie"

@app.route('/recommend/<movie_id>/')
def content(movie_id):
    resultat = utils.recom_movie(movie_id)
    return '%s' % resultat

if __name__ == "__main__":
    app.run()
