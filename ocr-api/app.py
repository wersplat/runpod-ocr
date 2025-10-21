from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from paddleocr import PaddleOCR
import numpy as np
import cv2, os, time

USE_GPU = os.getenv("PADDLE_OCR_USE_GPU", "1") == "1"
LANG = os.getenv("PADDLEOCR_LANG", "en")
MODEL_DIR = os.getenv("PADDLEOCR_MODEL_DIR", None)

app = FastAPI(title="PaddleOCR API", version="1.0")
_ocr = None

def get_ocr():
    global _ocr
    if _ocr is None:
        t0 = time.time()
        kwargs = dict(use_angle_cls=True, use_gpu=USE_GPU, lang=LANG)
        if MODEL_DIR and os.path.isdir(MODEL_DIR):
            kwargs.update(dict(det_model_dir=os.path.join(MODEL_DIR, "det"),
                               rec_model_dir=os.path.join(MODEL_DIR, "rec"),
                               cls_model_dir=os.path.join(MODEL_DIR, "cls")))
        _ocr = PaddleOCR(**kwargs)
        print(f"[OCR] Ready in {time.time()-t0:.2f}s (GPU={USE_GPU})")
    return _ocr

class OCRResponse(BaseModel):
    lines: list[str]
    boxes: list  # list of 4-point polygons
    confidences: list[float]

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/ocr", response_model=OCRResponse)
async def ocr(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image/* file")
    data = await file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image bytes")

    ocr = get_ocr()
    results = ocr.ocr(img, cls=True)

    lines, boxes, confs = [], [], []
    for res in results:
        for poly, (txt, conf) in res:
            lines.append(txt)
            boxes.append(poly)            # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            confs.append(float(conf))

    return OCRResponse(lines=lines, boxes=boxes, confidences=confs)