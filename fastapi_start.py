from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import torch
from torchvision.models import efficientnet_b4
from torchvision.transforms import functional as F
from model.efficient_net_modified import EfficientNetModified, load_efficientnetmodified
from dataset import label

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = load_efficientnetmodified("checkpoints/new-eff/new-eff_11.pth")
    models["efficientnet"] = model
    yield models
    models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/images/")
async def create_upload_file(file: UploadFile = File(...)):

    contents = await file.read()  # <-- Important!

    # Convert from bytes to PIL image
    img = Image.open(BytesIO(contents)).convert("RGB")
    
    # transform image
    img_tensor: torch.Tensor = EfficientNetModified.transform(img) # type: ignore
    
    # predict
    model = models["efficientnet"]
    model.eval()
    with torch.no_grad():
        preds = model(img_tensor.unsqueeze(0))
        topk_index = torch.topk(preds, 5, 1).indices.squeeze(0)
        print(topk_index)
        # probas = preds.softmax(1)
        # preds = probas.argmax(1)
        
    pred_summary = {}
    for idx in topk_index:
        class_id = idx.item()
        score = preds[0][class_id].item()
        class_name = label.GameLabel(class_id).name
        pred_summary[class_name] = score
        
    return {"filename": file.filename, "prediction": pred_summary}