from fastapi import FastAPI
from app.container import build_container
from app.router import register_routes
from core.config import Config

app = FastAPI()
config = Config()
container = build_container(config)
register_routes(app, container)
