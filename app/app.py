from fastapi import FastAPI


def create_app():
    app = FastAPI()

    @app.get("/")
    def read_root():
        return {"Hello": "World"}

    return app
