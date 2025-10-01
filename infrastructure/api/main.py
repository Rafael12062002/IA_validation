from fastapi import FastAPI
from infrastructure.api.routes import router

app = FastAPI(title="API - Validação Documentos")

app.include_router(router)
