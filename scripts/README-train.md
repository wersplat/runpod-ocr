# Prepare recognition dataset

1) From PaddleLabel, export OCR annotations, or write a Python script to:
   - Read full-image OCR polygons + text.
   - Crop each word region -> `data/rec_dataset/train/images/XXXXX.jpg`
   - Append lines to `data/rec_dataset/train/rec_gt.txt`:
       images/XXXXX.jpg<TAB>TRANSCRIPTION

2) Make a small `val/` split the same way.

3) Run training inside the ocr-api container (has GPU libs):
   docker compose exec ocr-api bash -lc "bash /workspace/scripts/train_rec.sh"

4) After training, the best weights appear under:
   /workspace/models/rec_ppocrv4/best_accuracy

5) To use them at runtime, copy the `inference` dir or set:
   PADDLEOCR_MODEL_DIR=/models
   and place:
     /models/rec  (your fine-tuned recognition)
     /models/det  (optional custom detector)
     /models/cls  (optional classifier)