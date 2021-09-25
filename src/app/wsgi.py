from typing import Final

import fastapi

import src.app.views.routes


def main() -> fastapi.FastAPI:
    app: Final = fastapi.FastAPI(title="MLOps Basics App")
    src.app.views.routes.add_routes(app)

    return app
