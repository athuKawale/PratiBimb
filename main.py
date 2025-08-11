import uvicorn
import warnings
from fastapi import FastAPI
from routers import faceswap, videoswap
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from metadata import version, title, description
from roop.utilities import delete_temp_directory
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings("ignore", message="resource_tracker: There appear to be")

process_pool = ProcessPoolExecutor()

@asynccontextmanager
async def lifespan(app: FastAPI):
  
    delete_temp_directory()
    
    yield

    process_pool.shutdown(wait=True)
    
    delete_temp_directory()

app = FastAPI(
    title=title,
    description=description,
    version=version,
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
