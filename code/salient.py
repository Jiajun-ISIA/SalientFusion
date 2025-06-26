# Imports
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

# Load BiRefNet with weights
from transformers import AutoModelForImageSegmentation



birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
# birefnet = ... # -- BiRefNet should be loaded with codes above, either way.
torch.set_float32_matmul_precision(['high', 'highest'][0])
birefnet.to('cuda')
birefnet.eval()

def extract_object(birefnet, imagepath):
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(imagepath)
    input_images = transform_image(image).unsqueeze(0).to('cuda')

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    return image, mask

# Visualization
plt.axis("off")
plt.imshow(extract_object(birefnet, imagepath='/data1/sjj/dataset_food/food101/images/apple_pie/68383.jpg')[0])
plt.savefig('example.jpg')
plt.show()
