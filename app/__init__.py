from flask import Flask
from flask_socketio import SocketIO

socketio = SocketIO()

def create_app(debug=False):
    app = Flask(__name__)
    app.debug = debug
    app.config['SECRET_KEY'] = 'your_secret_key_here'

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    socketio.init_app(app)
    return app
