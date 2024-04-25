import gradio as gr
from PIL import Image
import numpy as np
import torch
from dataset import label

from model.efficient_net_modified import EfficientNetModified, load_efficientnetmodified

model = load_efficientnetmodified("checkpoints/new-eff/new-eff_11.pth")

def greet(image, topk):
    # img = Image.open(image).convert("RGB")
    img = Image.fromarray(np.uint8(image)).convert('HSV')
    
    img_tensor: torch.Tensor = EfficientNetModified.transform(img)  # type: ignore
    model.eval()
    with torch.no_grad():
        preds = model(img_tensor.unsqueeze(0))
        topk_index = torch.topk(preds, topk, 1).indices.squeeze(0)
        print(topk_index)
        
    pred_summary = []
    for idx in topk_index:
        class_id = idx.item()
        score = preds[0][class_id].item()
        class_name = label.GameLabel(class_id).name
        pred_summary.append([class_name,score])
        
    return pred_summary[0][0], pred_summary

demo = gr.Interface(
    fn=greet,
    inputs=[gr.Image(), gr.Slider(minimum=1,maximum=30,value=5,step=1)],
    outputs=[gr.Textbox(label="Predicted class", lines=3), gr.Dataframe(headers=["classes","score"])],
    allow_flagging="never"
)

demo.launch()