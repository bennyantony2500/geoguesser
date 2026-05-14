import asyncio
from PIL import Image, ImageDraw, ImageFont
from model import predict_streetclip, extract_text_from_image, search_country_for_text, fuse_predictions

# ── Step 1: Create a test image with text on it ──────────────────────────────
print("Creating test image with street sign text...")
img = Image.new('RGB', (640, 480), color=(135, 206, 235))
draw = ImageDraw.Draw(img)

# Sky
draw.rectangle([0, 0, 640, 300], fill=(135, 206, 235))
# Road
draw.rectangle([0, 350, 640, 480], fill=(50, 50, 50))
# Street sign board
draw.rectangle([180, 150, 460, 230], fill=(255, 255, 255), outline=(0,0,0), width=3)
# Text on sign
draw.text((200, 165), "Rue de Rivoli", fill=(0, 0, 0))
draw.text((200, 195), "Paris, France",  fill=(0, 0, 0))

img.save('full_test.jpg')
print("✅ Test image created: full_test.jpg")
print()

# ── Step 2: Test StreetCLIP ───────────────────────────────────────────────────
print("=" * 50)
print("TEST 1 — StreetCLIP Visual Model")
print("=" * 50)
clip_results = predict_streetclip(img, top_k=5)
for p in clip_results:
    bar = '#' * int(p['confidence'])
    print(f"  {p['country']:<20} {p['confidence']:>6.2f}%  {bar}")
print()

# ── Step 3: Test EasyOCR ─────────────────────────────────────────────────────
print("=" * 50)
print("TEST 2 — EasyOCR Text Extraction")
print("=" * 50)
ocr_texts = extract_text_from_image(img)
if ocr_texts:
    print(f"  ✅ Text found: {ocr_texts}")
else:
    print("  ⚠ No text detected (try a real street image)")
print()

# ── Step 4: Test Web Search ───────────────────────────────────────────────────
print("=" * 50)
print("TEST 3 — Web Search for OCR text")
print("=" * 50)
async def test_web():
    if ocr_texts:
        result = await search_country_for_text(ocr_texts)
        if result:
            print(f"  ✅ Country found : {result['country']}")
            print(f"  Confidence       : {result['confidence']}")
            print(f"  Searched text    : {result['source_text']}")
            print(f"  Snippet          : {result['search_snippet'][:100]}")
        else:
            print("  ⚠ No country matched from web search")
    else:
        print("  ⚠ Skipped — no OCR text to search")
    return await search_country_for_text(ocr_texts) if ocr_texts else None

ocr_result = asyncio.run(test_web())
print()

# ── Step 5: Test Ensemble Fusion ──────────────────────────────────────────────
print("=" * 50)
print("TEST 4 — Ensemble Fusion (all signals combined)")
print("=" * 50)
final = fuse_predictions(clip_results, ocr_result, ocr_texts)

print(f"  🌍 Top Guess  : {final['top_guess']}")
print(f"  🔤 OCR Match  : {final['ocr_country'] or 'None'}")
print(f"  📝 OCR Texts  : {final['ocr_texts']}")
print()
print("  Final Rankings:")
for p in final['predictions']:
    bar = '#' * int(p['confidence'] / 2)
    boosted = " ← OCR boosted" if p['country'] == final['ocr_country'] else ""
    print(f"  {p['country']:<20} {p['confidence']:>6.2f}%  {bar}{boosted}")