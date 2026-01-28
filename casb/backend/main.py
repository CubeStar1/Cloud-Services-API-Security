from fastapi import FastAPI

from .routes import router as api_router

app = FastAPI(title="Cloud Services API Security Backend")

app.include_router(api_router)


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
