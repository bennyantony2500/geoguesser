from PIL import Image
from model import predict_streetclip

img = Image.open('street.png').convert('RGB')
print('Image size:', img.size)

results = predict_streetclip(img, top_k=5)

print()
print('===== StreetCLIP Results =====')
for p in results:
    bar = '#' * int(p['confidence'])
    print(f"{p['country']:<20} {p['confidence']:>6.2f}%  {bar}")