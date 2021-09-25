import fastapi

import src.app.views.api


def add_routes(app: fastapi.FastAPI) -> None:
    app.add_api_route(
        "/health",
        src.app.views.api.health,
        methods=["GET"],
    )

    app.add_api_route(
        "/predict",
        src.app.views.api.predict,
        methods=["POST"],
    )

    app.add_api_route(
        "/predict/label",
        src.app.views.api.predict_label,
        methods=["POST"],
    )
