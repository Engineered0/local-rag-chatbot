import base64, io, ollama, pytesseract
from PIL import Image
from pdf2image import convert_from_path
from config import VISION_MODEL

_cache = {}
_vision_ok = None

MAX_SIDE = 1024          # moondream chokes on full-res photos
BAD = {"!!!image!!!", "!!!IMAGE!!!", "", "."}


def render(path, pno, dpi):
    """A PDF page, or an image file directly (pno=0)."""
    key = (path, pno, dpi)
    if key not in _cache:
        if pno == 0:
            _cache[key] = Image.open(path).convert("RGB")
        else:
            _cache[key] = convert_from_path(path, first_page=pno,
                                            last_page=pno, dpi=dpi)[0]
    return _cache[key]


def ocr(path, pno=0, dpi=400):
    try:
        return pytesseract.image_to_string(render(path, pno, dpi)).strip()
    except Exception as e:
        print(f"\n  ocr failed p{pno}: {e}")
        return ""


def vision_ready():
    """Checked once per run."""
    global _vision_ok
    if _vision_ok is None:
        try:
            names = [m.model for m in ollama.list().models]
            _vision_ok = any(n.startswith(VISION_MODEL) for n in names)
        except Exception:
            _vision_ok = False
        if not _vision_ok:
            print(f"  (vision off — run: ollama pull {VISION_MODEL})")
    return _vision_ok


def _encode(img):
    """RGB, downscaled, JPEG, base64 — what the vision model expects."""
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def describe(path, pno=0):
    if not vision_ready():
        return ""
    try:
        b64 = _encode(render(path, pno, 200))
        r = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": "Describe this image. Include any text, labels, "
                           "or numbers you can read.",
                "images": [b64],
            }],
        )
        out = r["message"]["content"].strip()
        if out.lower() in {b.lower() for b in BAD} or len(out) < 15:
            print(f"\n  vision returned nothing usable p{pno}: {out[:40]!r}")
            return ""
        return out
    except Exception as e:
        print(f"\n  vision failed p{pno}: {e}")
        return ""

def _raw(path, pno=0):
    b64 = _encode(render(path, pno, 200))
    r = ollama.chat(
        model=VISION_MODEL,
        messages=[{"role": "user",
                   "content": "What is in this image?",
                   "images": [b64]}],
    )
    print(r)


def clear_cache():
    _cache.clear()


if __name__ == "__main__":
    import sys
    p = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    im = render(p, n, 200)
    print(f"mode={im.mode} size={im.size}")
    print("--- ocr ---")
    print(ocr(p, n)[:400] or "(nothing)")
    print("--- vision ---")
    print(describe(p, n) or "(nothing)")