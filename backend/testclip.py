from PIL import Image, ImageDraw
from model import predict_streetclip

# Create a simple test image
img = Image.new('RGB', (640, 480), color=(135, 206, 235))
draw = ImageDraw.Draw(img)
draw.rectangle([0, 350, 640, 480], fill=(50, 50, 50))
img.save('clip_test.jpg')
print('Image created...')

# Run StreetCLIP
print('Running StreetCLIP...')
results = predict_streetclip(img, top_k=5)

print()
print('===== StreetCLIP Results =====')
for p in results:
    bar = '#' * int(p['confidence'] / 2)
    print(f"{p['country']:<20} {p['confidence']:>6.2f}%  {bar}")