GeoGuesser AI

An AI-powered country guesser that analyzes street view images using an ensemble of three models — visual recognition, text extraction, and web search — to predict which country a photo was taken in.

**Live App:** [bennyantony2500-geoguesser.hf.space/app](https://bennyantony2500-geoguesser.hf.space/app)


## Working

The project uses an ensemble of methods. It fuses their result and gives the list of countries.Upload any street view image and the ensemble pipeline runs three models in parallel, fuses their signals, and returns a ranked list of country predictions with confidence scores.


## Models

### 1. StreetCLIP — Visual Country Recognition

**Model:** [`geolocal/StreetCLIP`](https://huggingface.co/geolocal/StreetCLIP)  
**Type:** CLIP-based zero-shot image classifier  
**Size:** ~1.7GB  

StreetCLIP is a CLIP model fine-tuned specifically on street-level imagery from around the world. It was trained on millions of Google Street View images paired with their geographic locations.

**How it works:**
- Takes the uploaded image and generates text prompts for every country: `"a street view of France"`, `"a street view of Japan"`, etc.


### 2. EasyOCR — Text Extraction

**Library:** [`JaidedAI/EasyOCR`](https://github.com/JaidedAI/EasyOCR)  
**Type:** Deep learning OCR (CRAFT + CRNN)  
**Languages supported:** English, French, German, Spanish, Italian, Portuguese, Dutch, Polish, Turkish  

EasyOCR uses two neural networks working together to find and read text in images.

**How it works:**
- **CRAFT (Character Region Awareness For Text detection):** A convolutional neural network that finds regions of the image likely to contain text by generating heatmaps around individual characters and words
- **CRNN (Convolutional Recurrent Neural Network):** Reads the detected text regions and converts them to strings using CTC (Connectionist Temporal Classification) decoding


### 3. Web Search — Location Lookup

**API:** Wikipedia Search API (free, no key required)  
**Fallback:** DuckDuckGo HTML search  

Once EasyOCR extracts text from the image, the web search model tries to identify which country that text belongs to.

**How it works:**
1. **Direct match:** Checks if any extracted text directly contains a country name (e.g. `"Paris, France"` → France)
2. **Wikipedia search:** Searches each OCR token individually against Wikipedia's search API and scans results for country name mentions
3. **DuckDuckGo fallback:** If Wikipedia returns nothing useful, scrapes DuckDuckGo HTML results for country mentions

**Example:**
```
OCR finds:  ["Rue de Rivoli", "Paris"]
Searches:   "Rue de Rivoli city country" on Wikipedia
Finds:      "Rue de Rivoli is a street in Paris, France..."
Returns:    France (confidence: 65%)
```



## Fusion

The three models are combined using a weighted scoring system:

| Signal | Weight |
|---|---|
| StreetCLIP visual prediction | 55% |
| OCR + Web Search result | 45% |

**Fusion logic:**
1. StreetCLIP returns top-5 countries with raw confidence scores, weightage is 0.55
2. If the web search finds a country, that country's score gets an additional boost of weightage is 0.45
3. All scores are re-normalised to sum to 100%
4. The top-5 final predictions are returned



## Countries Supported

[
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

---

## Local Development
```bash
# Clone the repo
git clone https://github.com/bennyantony2500/geoguesser
cd geoguesser/backend

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload

# Open the frontend
# Open frontend/index.html with Live Server in VS Code
```

