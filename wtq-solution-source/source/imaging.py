"""Load a bill image, fix orientation, downscale if huge, return base64 JPEG."""
import base64
import io

from PIL import Image, ImageOps

MAX_SIDE = 2400


def load_image_b64(path: str) -> tuple[str, str]:
    """Returns (mime_type, base64_data)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        w, h = im.size
        scale = MAX_SIDE / max(w, h)
        if scale < 1:
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
    return "image/jpeg", base64.b64encode(buf.getvalue()).decode("ascii")
