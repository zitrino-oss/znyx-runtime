"""Bounded, offline OCR and transcription. Optional dependencies, never stub extraction."""
from __future__ import annotations

import base64
from contextlib import closing
import functools
import io
import os

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

MAX_BYTES = 8 * 1024 * 1024
MAX_TEXT = 1024 * 1024
MAX_PIXELS = 12_000_000
router = APIRouter()


class MediaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: str = Field(max_length=4 * ((MAX_BYTES + 2) // 3))
    mime_type: str = Field(max_length=100)


@functools.lru_cache(maxsize=1)
def speech_model():
    path = os.getenv("ZNYX_SPEECH_MODEL_PATH", "")
    if not path or not os.path.isdir(path):
        raise RuntimeError("Install an offline speech model and set ZNYX_SPEECH_MODEL_PATH")
    import hashlib
    import re
    expected = os.getenv("ZNYX_SPEECH_MODEL_SHA256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise RuntimeError("Set ZNYX_SPEECH_MODEL_SHA256 to the approved model.bin digest")
    digest = hashlib.sha256()
    with open(os.path.join(path, "model.bin"), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError("Speech model digest differs from the approved version")
    from faster_whisper import WhisperModel
    return WhisperModel(path, device="cpu", compute_type="int8", local_files_only=True, cpu_threads=2)


def extract(data: bytes, mime_type: str) -> dict:
    method = ""
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Media must contain 1..8388608 bytes")
    if mime_type.startswith("text/") or mime_type in ("application/json", "application/xml"):
        text, method = data.decode("utf-8"), "utf8"
    elif mime_type.startswith("image/"):
        from PIL import Image
        import pytesseract
        Image.MAX_IMAGE_PIXELS = MAX_PIXELS
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > MAX_PIXELS or getattr(image, "n_frames", 1) != 1:
                raise ValueError("Image is too large or animated")
            image.load()
            text, method = pytesseract.image_to_string(image, timeout=15), "tesseract"
    elif mime_type == "application/pdf":
        import pypdfium2 as pdfium
        import pytesseract
        with pdfium.PdfDocument(data) as document:
            if not 1 <= len(document) <= 8:
                raise ValueError("PDF exceeds 8 pages")
            pages = []
            for index in range(len(document)):
                # OCR the rendered page as well as reading its text layer: hidden text,
                # flattened images and scanned pages all reach the guard.
                with closing(document[index]) as page:
                    width, height = page.get_size()
                    if width * height * 4 > MAX_PIXELS:
                        raise ValueError("PDF page exceeds the render limit")
                    with closing(page.get_textpage()) as textpage:
                        pages.append(textpage.get_text_bounded())
                    with closing(page.render(scale=2)) as bitmap:
                        with bitmap.to_pil() as image:
                            pages.append(pytesseract.image_to_string(image, timeout=15))
            text, method = "\n".join(pages), "pdf-text-and-ocr"
    elif mime_type in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       "application/vnd.openxmlformats-officedocument.presentationml.presentation"):
        import mimetypes
        import zipfile
        from defusedxml import ElementTree
        chunks, images = [], 0
        with zipfile.ZipFile(io.BytesIO(data)) as document:
            entries = document.infolist()
            if len(entries) > 1000 or sum(entry.file_size for entry in entries) > MAX_BYTES:
                raise ValueError("Office document exceeds the expansion limit")
            for entry in entries:
                if entry.is_dir():
                    continue
                if "/embeddings/" in entry.filename or entry.filename.lower().endswith(".bin"):
                    raise ValueError("Embedded objects and macros cannot be completely inspected")
                if entry.filename.endswith((".xml", ".rels")):
                    raw = document.read(entry)
                    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
                        raise ValueError("XML entities are refused")
                    root = ElementTree.fromstring(raw, forbid_dtd=True, forbid_entities=True, forbid_external=True)
                    chunks.append("".join(root.itertext()))
                    chunks.append("\n".join(root.itertext()))
                    chunks.extend(value for element in root.iter() for value in element.attrib.values()
                                  if value.startswith(("http://", "https://")))
                elif "/media/" in entry.filename:
                    images += 1
                    if images > 32:
                        raise ValueError("Office document exceeds 32 embedded images")
                    image_type = mimetypes.guess_type(entry.filename)[0] or "application/octet-stream"
                    chunks.append(extract(document.read(entry), image_type)["text"])
                if sum(len(chunk) for chunk in chunks) > MAX_TEXT:
                    raise ValueError("Office document text exceeds the inspection limit")
        text, method = "\n".join(chunks), "office-xml-and-ocr"
    elif mime_type.startswith("audio/"):
        import av
        with av.open(io.BytesIO(data)) as container:
            if len(container.streams.audio) != 1 or container.streams.video:
                raise ValueError("Supply one audio track without video")
            duration = 0.0
            # Container duration can be missing or forged. Count decoded samples,
            # stopping at the limit, so compressed audio cannot bypass the bound.
            for frame in container.decode(audio=0):
                duration += frame.samples / frame.sample_rate
                if duration > 120:
                    raise ValueError("Audio exceeds 120 seconds")
            if duration <= 0:
                raise ValueError("Audio contains no samples")
        segments, _ = speech_model().transcribe(io.BytesIO(data), beam_size=1, vad_filter=False)
        text, method = " ".join(segment.text for segment in segments), "offline-whisper"
    else:
        raise ValueError("Unsupported media type")
    if len(text) > MAX_TEXT:
        raise ValueError("Extracted text exceeds 1 MiB")
    return {"text": text, "method": method, "complete": True,
            **({"duration_seconds": duration} if mime_type.startswith("audio/") else {})}


@router.post("/v1/media/extract")
async def media_extract(body: MediaRequest, request: Request):
    import asyncio
    import concurrent.futures
    import multiprocessing
    # Processes bound both concurrency and memory lifetime; a timed-out caller does not
    # free its slot until extraction really finishes.
    if not hasattr(request.app.state, "media_pool"):
        request.app.state.media_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=2, mp_context=multiprocessing.get_context("spawn"))
        request.app.state.media_slots = asyncio.Semaphore(2)
    slots = request.app.state.media_slots
    if slots.locked():
        raise HTTPException(429, "Media extraction is busy")
    try:
        data = base64.b64decode(body.data, validate=True)
    except ValueError:
        raise HTTPException(422, "Malformed base64 media") from None
    await slots.acquire()
    future = asyncio.get_running_loop().run_in_executor(request.app.state.media_pool, extract, data, body.mime_type)
    future.add_done_callback(lambda _: slots.release())
    try:
        return await asyncio.wait_for(asyncio.shield(future), timeout=150)
    except (ImportError, RuntimeError):
        raise HTTPException(503, "Media extractor is unavailable; install the media extra and offline models") from None
    except (ValueError, UnicodeError):
        raise HTTPException(422, "Media could not be completely inspected") from None
    except TimeoutError:
        raise HTTPException(504, "Media extraction timed out") from None
    except Exception:
        raise HTTPException(422, "Invalid or unsupported media") from None
