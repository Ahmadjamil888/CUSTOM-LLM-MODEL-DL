from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable Cross-Origin if needed (for frontend integration)

    # Register routes
    from .routes import main
    app.register_blueprint(main)

    return app
