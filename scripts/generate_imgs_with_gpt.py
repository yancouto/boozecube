from openai import OpenAI
from pathlib import Path
import base64
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed

client = OpenAI()

IN = Path("prompts_gpt/")
ORIG = Path("../boozecube.mse-set/")
interrupted = False
DALLE = False
if DALLE:
    OUT = Path("out_gpt_dalle3/")
else:
    OUT = Path("out_gpt_gpt-image/")

OUT.mkdir(parents=True, exist_ok=True)


def handle_interrupt(signal, frame):
    print("\nFinishing current image before exiting...", flush=True)
    global interrupted
    interrupted = True


signal.signal(signal.SIGINT, handle_interrupt)


def process_file(file):
    global interrupted
    if interrupted or (OUT / (file.name + ".png")).exists():
        return False
    # Horizontal by default
    size = "1792x1024" if DALLE else "1536x1024"
    original = (ORIG / file.name).read_text(encoding="utf-8")
    if (
        ("super_type: <word-list-type-en>Token" in original)
        or ("super_type: <word-list-type-en>Emblem" in original)
        or ("super_type: <word-list-type-en>Trap" in original)
        or ("sub_type: <word-list-enchantment>Saga" in original)
    ):
        size = "1024x1792" if DALLE else "1024x1536"

    print("Creating image for: " + file.name[5:], flush=True)
    try:
        response = client.images.generate(
            model="dall-e-3" if DALLE else "gpt-image-1",
            prompt=file.read_text(encoding="utf-8")
            + " Don't show the actual card, just the art.",
            n=1,
            size=size,
            quality="hd" if DALLE else "high",
            **({"response_format": "b64_json"} if DALLE else {"background": "opaque"}),  # type: ignore
        )
        assert response.data, "Request failed"
        img = response.data[0].b64_json
        assert img is not None, "Image not found in response"
        image_bytes = base64.b64decode(img)
        (OUT / (file.name + ".png")).write_bytes(image_bytes)
    except Exception as e:
        print(f"Error generating image for {file.name}: {e}", flush=True)
        interrupted = True
        return False
    return True


max_images = 1000

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(process_file, file) for file in IN.iterdir()]
    for future in as_completed(futures):
        if future.result():
            max_images -= 1
        if interrupted or max_images <= 0:
            break
    executor.shutdown(cancel_futures=True, wait=True)
