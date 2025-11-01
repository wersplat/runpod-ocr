# Training Documentation

## Deprecated: PaddleOCR Training (Legacy)

**Note:** This project has been migrated to DeepSeek-OCR. The following PaddleOCR training instructions are kept for reference only.

### Prepare recognition dataset

1) From PaddleLabel, export OCR annotations, or write a Python script to:
   - Read full-image OCR polygons + text.
   - Crop each word region -> `data/rec_dataset/train/images/XXXXX.jpg`
   - Append lines to `data/rec_dataset/train/rec_gt.txt`:
       images/XXXXX.jpg<TAB>TRANSCRIPTION

2) Make a small `val/` split the same way.

3) Run training inside the ocr-api container (has GPU libs):
   ```bash
   docker compose exec ocr-api bash -lc "bash /workspace/scripts/train_rec.sh"
   ```

4) After training, the best weights appear under:
   `/workspace/models/rec_ppocrv4/best_accuracy`

5) To use them at runtime, copy the `inference` dir or set:
   `PADDLEOCR_MODEL_DIR=/models`
   and place:
     `/models/rec`  (your fine-tuned recognition)
     `/models/det`  (optional custom detector)
     `/models/cls`  (optional classifier)

---

## DeepSeek-OCR Fine-tuning

DeepSeek-OCR is a 3B parameter vision-language model optimized for document OCR. For fine-tuning instructions and capabilities, please refer to:

- **Official GitHub Repository**: [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- **Hugging Face Model Card**: [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- **Research Paper**: [arXiv:2510.18234](https://arxiv.org/abs/2510.18234)

### Key Features:
- Supports context-aware OCR with markdown conversion
- Optimized for document understanding
- Can be fine-tuned for domain-specific tasks
- Supports various size modes (Tiny, Small, Base, Large, Gundam)

### Training Infrastructure:
The training infrastructure (`scripts/train_rec.sh`) is preserved for potential future use with DeepSeek-OCR fine-tuning, but currently requires adaptation to the DeepSeek-OCR training framework.
