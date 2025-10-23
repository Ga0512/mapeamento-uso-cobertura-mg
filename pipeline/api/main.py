from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pipeline.api.routes.predict import router

app = FastAPI(
    title="IEF",
    version="1.0.0",
    description="API para segmentacao de imagens de satélites"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

app.include_router(router)

@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}
