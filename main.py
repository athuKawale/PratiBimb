import uvicorn
import warnings
from fastapi import FastAPI
from routers import faceswap, videoswap
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from roop.utilities import delete_temp_directory
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore", message="resource_tracker: There appear to be")

# Create a global variable for your pool
process_pool = ProcessPoolExecutor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Before the application starts, initialize resources
    delete_temp_directory()
    
    yield

    process_pool.shutdown(wait=True)  # Ensures all processes and semaphores are cleaned
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
    try:
        uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
    except Exception as e:
        print("Error \n", e)
        process_pool.shutdown(wait=True)
        print("Process pool shutdown complete. Exiting.")
