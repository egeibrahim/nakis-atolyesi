"""Nakış Atölyesi profesyonel motor: görsel + ayarlar -> Ink/Stitch PES/DST."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
import asyncio

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from digitize import digitize

app = FastAPI(title="Nakış Atölyesi")
origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["POST", "GET"], allow_headers=["*"])
pool = ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", "2")))
MAX_BYTES = 12 * 1024 * 1024


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/digitize")
async def api_digitize(image: UploadFile = File(...), params: str = Form("{}")):
    data = await image.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Görsel 12 MB'den büyük olamaz.")
    try:
        prm = json.loads(params)
    except json.JSONDecodeError:
        raise HTTPException(400, "Ayarlar okunamadı.")
    try:
        return await asyncio.get_running_loop().run_in_executor(pool, digitize, data, prm)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:  # Ink/Stitch or geometry failure
        raise HTTPException(500, f"Dikiş üretilemedi: {e}")
