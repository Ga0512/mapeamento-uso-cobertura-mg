from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import List
import os
import uuid
import shutil
from src.models import SegmentationModels
from pipeline.api.core.logger import logger

import os
from fastapi import FastAPI

CI_MODE = os.getenv("CI", "false").lower() == "true"

app = FastAPI()

if CI_MODE:
    from tests.mocks import MockModel
    model = MockModel()
else:
    from src.models import SegmentationModels
    model = SegmentationModels()

router = APIRouter()

@router.post("/predict", tags=["Predict"])
async def predict(
    images: List[UploadFile] = File(...),
    model_name: str = Form(...)
):
    logger.info(f"Recebido {len(images)} imagens para modelo {model_name}")
    masks = [] 

    for image in images:
        if not image.content_type or not image.content_type.startswith("image/"):
            logger.error(f"Erro ao processar {image.filename}")
            raise HTTPException(
                status_code=400,
                detail=f"{image.filename} não é uma imagem válida."
            )

        temp_name = f"{uuid.uuid4()}_{image.filename}"   
        temp_path = os.path.join("temp", temp_name)

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        if model_name == "deeplab":
            mask = model.deeplab.predict(temp_path)
        elif model_name == "segformer":
            mask = model.segformer.predict(temp_path)
        else:
            mask = model.unet.predict(temp_path)

        masks.append(mask.tolist())  

        try:
            os.remove(temp_path)
        except:
            pass
    
    logger.info(f"Retornando {len(masks)} máscaras")
    return {"masks": masks}
