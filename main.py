from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
import io

app = FastAPI(title="Visual Question Answering API")

# Load the pre-trained model and processor
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

@app.post("/vqa/")
async def visual_question_answer(image: UploadFile = File(...), question: str = "What is in the image?"):
    try:
        # Read and process the image
        image_data = await image.read()
        image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Prepare inputs for the model
        encoding = processor(image_pil, question, return_tensors="pt")

        # Perform inference
        outputs = model(**encoding)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        answer = model.config.id2label[predicted_class_idx]

        return {"question": question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
def health_check():
    return {"status": "healthy"}
