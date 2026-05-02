from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
import gc
 
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cpu"

# =========================
# CLIP MODEL (LAZY LOAD)
# =========================
clip_model = None
clip_preprocess = None


def load_clip():
    global clip_model, clip_preprocess

    if clip_model is None:
        import clip
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        clip_model.eval()

    return clip_model, clip_preprocess


def unload_clip():
    global clip_model, clip_preprocess
    clip_model = None
    clip_preprocess = None
    gc.collect()


# =========================
# EMBEDDING FUNCTION
# =========================
def get_embedding(image: Image.Image):
    model, preprocess = load_clip()

    image = image.convert("RGB")

    # =========================
    # 🔥 MULTI-CROP
    # =========================
    img1 = preprocess(image).unsqueeze(0)
    img2 = preprocess(image.resize((256, 256))).unsqueeze(0)

    with torch.no_grad():
        v1 = model.encode_image(img1)
        v2 = model.encode_image(img2)

    # =========================
    # ✅ NORMALIZE
    # =========================
    v1 = v1 / v1.norm(dim=-1, keepdim=True)
    v2 = v2 / v2.norm(dim=-1, keepdim=True)

    # =========================
    # 🔥 FINAL VECTOR
    # =========================
    vec = (v1 + v2) / 2

    result = vec.squeeze().tolist()

    # =========================
    # 🧹 MEMORY CLEANUP
    # =========================
    del img1, img2, v1, v2, vec
    gc.collect()

    # ⚠️ VERY IMPORTANT FOR 512MB
    unload_clip()

    return result


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {
        "status": "API is running 🚀",
        "model": "CLIP ViT-B/32 (multi-crop optimized)"
    }


# =========================
# IMAGE VECTOR API
# =========================
@app.post("/embed-image")
async def embed_image(file: UploadFile = File(...)):

    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        return {"error": "Only image files allowed"}

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        vector = get_embedding(image)

        return {
            "embedding": vector,
            "dimension": len(vector),   # 512
            "model": "clip-vit-b32-multicrop"
        }

    except Exception as e:
        return {"error": str(e)}
