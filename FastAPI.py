from fastapi import FastAPI
from pydantic import BaseModel
import base64
from ultralytics import YOLO
import os
import numpy as np
import cv2


HOME = os.getcwd()
model = YOLO('best.pt')

class Item(BaseModel):
    image : str
app =FastAPI()

@app.post("/items/")
async def post_item(item : Item):
    image_data = base64.b64decode(item.image)
    array_np=np.frombuffer(image_data,dtype=np.uint8)
    img = cv2.imdecode(array_np, cv2.IMREAD_COLOR)
    result = model(img)[0]
    json_data=result.tojson()
    payload={
        "image":json_data
    }
    return payload
    #detections = sv.Detections.from_ultralytics(result)
    # bounding_box_annotator = sv.BoundingBoxAnnotator()
    # annotated_frame = bounding_box_annotator.annotate(
    #     scene=array_np,
    #     detections=detections
    # )
    # bytes_image=annotated_frame.tobytes()
    # encoded_image = base64.b64encode(bytes_image).decode('utf-8')
    # payload = {
    #     "image": encoded_image
    # }
    # return payload


@app.get("/")
async def index():
    return "Hello World"