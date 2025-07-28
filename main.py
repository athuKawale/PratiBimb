import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routers import faceswap, videoswap

app = FastAPI(
    title="Face Swap API",
    description="An API for performing face swaps and managing templates.",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")
# Register routers
app.include_router(faceswap.router)
app.include_router(videoswap.router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)