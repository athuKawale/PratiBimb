import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from roop.utilities import delete_temp_directory
from routers import faceswap, videoswap

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Before the application starts, initialize resources

    yield

    # After the application stops, clean up resources

    delete_temp_directory()

app = FastAPI(
    title="Face Swap API",
    description="An API for performing face swaps and managing templates.",
    version="1.0.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Register routers
app.include_router(faceswap.router)
app.include_router(videoswap.router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
