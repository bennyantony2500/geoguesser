import torch
import easyocr
import httpx
import re
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


COUNTRIES = [
    "Afghanistan", "Albania", "Algeria", "Argentina", "Australia", "Austria",
    "Bangladesh", "Belgium", "Bolivia", "Brazil", "Bulgaria", "Cambodia",
    "Canada", "Chile", "China", "Colombia", "Croatia", "Czech Republic",
    "Denmark", "Ecuador", "Egypt", "Estonia", "Ethiopia", "Finland",
    "France", "Germany", "Ghana", "Greece", "Guatemala", "Hungary",
    "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy",
    "Japan", "Jordan", "Kazakhstan", "Kenya", "Kyrgyzstan", "Laos",
    "Latvia", "Lebanon", "Lithuania", "Madagascar", "Malaysia", "Mexico",
    "Mongolia", "Morocco", "Mozambique", "Myanmar", "Nepal", "Netherlands",
    "New Zealand", "Nigeria", "Norway", "Pakistan", "Panama", "Paraguay",
    "Peru", "Philippines", "Poland", "Portugal", "Romania", "Russia",
    "Rwanda", "Saudi Arabia", "Senegal", "Serbia", "Singapore", "Slovakia",
    "South Africa", "South Korea", "Spain", "Sri Lanka", "Sweden",
    "Switzerland", "Taiwan", "Tanzania", "Thailand", "Tunisia", "Turkey",
    "Uganda", "Ukraine", "United Kingdom", "United States", "Uruguay",
    "Uzbekistan", "Venezuela", "Vietnam", "Zambia", "Zimbabwe"
]


STREETCLIP_MODEL="geolocal/StreetCLIP"
print("loading streeclip")
clip_model=CLIPModel.from_pretrained(STREETCLIP_MODEL)
clip_processor=CLIPProcessor.from_pretrained(STREETCLIP_MODEL)
clip_model.eval()
print("StreetCLIP loaded")


print("Loading EasyOCR...")
_gpu = torch.cuda.is_available()

ocr_reader_latin = easyocr.Reader(
    ["en", "fr", "de", "es", "it", "pt", "nl", "pl", "tr"],
    gpu=_gpu,
    verbose=False,
)
print("EasyOCR loaded.")



def predict_streetclip(image:Image.Image, top_k: int=5)->list:
    text_prompts=[f"a street view of {c}" for c in COUNTRIES]
    inputs=clip_processor(
        text=text_prompts,images=image, return_tensors="pt",padding=True
    )

    with torch.no_grad():
        probs=clip_model(**inputs).logits_per_image.softmax(dim=1)[0]

    top_idx=probs.topk(top_k).indices.tolist()

    return [
        {"country":COUNTRIES[i], "confidence":round(float(probs[i])*100,4)}
        for i in top_idx
    ]


def extract_text_from_image(image: Image.Image) -> list:
    img_array = np.array(image)
    results = ocr_reader_latin.readtext(img_array, detail=1, paragraph=False)
    return [r[1].strip() for r in results if r[2] > 0.3 and len(r[1].strip()) > 1]

async def search_country_for_text(texts: list) -> dict | None:
    if not texts:
        return None

    headers = {"User-Agent": "GeoGuesserBot/1.0 (educational project)"}

    
    candidates = [t for t in texts if len(t.strip()) >= 3]
    if not candidates:
        return None

    print(f"[WEB] Searching for: {candidates}")

    
    all_text = " ".join(candidates).lower()
    direct = _find_country(all_text)
    if direct:
        print(f"[WEB] Direct country name found in image: {direct}")
        return {
            "country": direct,
            "confidence": 90.0,
            "source_text": " ".join(candidates),
            "search_snippet": f'Country name "{direct}" found directly in image text.',
        }

    
    for token in candidates[:6]:
        try:
            async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
                # Wikipedia opensearch — returns titles + descriptions
                resp = await client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": token,
                        "format": "json",
                        "srlimit": 3,
                        "srprop": "snippet",
                    },
                    headers=headers,
                )
                data = resp.json()

            results = data.get("query", {}).get("search", [])
            for r in results:
                combined = (r.get("title", "") + " " + r.get("snippet", "")).lower()
                # strip HTML tags from snippet
                combined = re.sub(r"<[^>]+>", "", combined)
                print(f"[WIKI] '{token}' → '{combined[:100]}'")

                found = _find_country(combined)
                if found:
                    print(f"[WIKI] Match: {found}")
                    return {
                        "country": found,
                        "confidence": 65.0,
                        "source_text": token,
                        "search_snippet": re.sub(r"<[^>]+>", "", r.get("snippet", ""))[:240],
                    }

        except Exception as exc:
            print(f"[WIKI] Error for '{token}': {exc}")
            continue

    
    full_query = " ".join(candidates[:3])
    try:
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "opensearch",
                    "search": full_query,
                    "limit": 5,
                    "format": "json",
                },
                headers=headers,
            )
            data = resp.json()

        
        titles = data[1] if len(data) > 1 else []
        descriptions = data[2] if len(data) > 2 else []
        combined = " ".join(titles + descriptions).lower()
        print(f"[WIKI] Opensearch '{full_query}' → '{combined[:120]}'")

        found = _find_country(combined)
        if found:
            return {
                "country": found,
                "confidence": 50.0,
                "source_text": full_query,
                "search_snippet": " | ".join(descriptions[:2])[:240],
            }

    except Exception as exc:
        print(f"[WIKI] Opensearch error: {exc}")

    
    try:
        query = " ".join(candidates[:3]) + " which country"
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
            resp = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers=headers,
            )
        found = _find_country(resp.text.lower())
        print(f"[DDG] HTML scrape → {found}")
        if found:
            return {
                "country": found,
                "confidence": 38.0,
                "source_text": query,
                "search_snippet": f'Inferred from web search: "{query}"',
            }

    except Exception as exc:
        print(f"[DDG] Error: {exc}")

    print(f"[WEB] No country found for: {candidates}")
    return None


def _find_country(text:str)->str|None:
    for co in COUNTRIES:
        if re.search(r"\b"+ re.escape(co.lower()) + r"\b", text):
            return co
        
    return None



def fuse_predictions(clip_preds:list, ocr_result:dict|None,ocr_texts:list)->dict:
    CLIP_W=0.55
    OCR_W=0.45


    scores:dict={}
    for p in clip_preds:
        scores[p["country"]]=p["confidence"]*CLIP_W

    if ocr_result:
        c=ocr_result["country"]
        scores[c]=scores.get(c,0) + ocr_result["confidence"]*OCR_W


    ranked=sorted(scores.items(), key=lambda x:x[1], reverse=True)
    top5=ranked[:5]
    total=sum(s for _, s in top5) or 1

    final_preds=[
        {"country":c, "confidence": round(s/total*100,2)}
        for c,s in top5
    ]


    return {
        "predictions":final_preds,
        "top_guess":final_preds[0]["country"] if final_preds else None,
        "ocr_texts": ocr_texts,
        "ocr_country":ocr_result["country"] if ocr_result else None,
        "ocr_snippet":ocr_result.get("search_snippet","") if ocr_result else None,
        "ocr_source_text": ocr_result.get("source_text","") if ocr_result else None
    }