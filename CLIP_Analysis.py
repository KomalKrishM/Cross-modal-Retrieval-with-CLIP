import numpy as np
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    processed_image = preprocess(image).unsqueeze(0).to(device)
    return image, processed_image

# Function to display an image
def display_image(image_path, title="Image"):
    image = plt.imread(image_path)
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

# Function to compute similarity between images and a text query
def image_retrieval(images, query_text):
    with torch.no_grad():
        image_features = torch.cat([model.encode_image(img) for img in images])
        text_features  = model.encode_text(query_text)
        print(image_features[0])
        print(text_features)
    similarity = (image_features @ text_features.T).softmax(dim=0)
    max_index  = similarity.argmax().item()
    return similarity, max_index

# Function to compute similarity between a query image and text descriptions
def text_retrieval(query_image, texts):
    with torch.no_grad():
        image_features = model.encode_image(query_image)
        text_features  = model.encode_text(texts)

    similarity = (image_features @ text_features.T).softmax(dim=-1)
    max_index  = similarity.argmax().item()
    return similarity, max_index


# Image paths
image_path1 = "./Cross_modal_Retrieval/6734417.jpg"

image_path2 = "./Cross_modal_Retrieval/8664922.jpg"

image_path3 = "./Cross_modal_Retrieval/19610188.jpg"

# Load and preprocess images
# display_image(image_path1, "Image 1")
image1, processed_image1 = load_and_preprocess_image(image_path1)

# display_image(image_path2, "Image 2")
image2, processed_image2 = load_and_preprocess_image(image_path2)

# display_image(image_path3, "Image 3")
image3, processed_image3 = load_and_preprocess_image(image_path3)

print("image shape:", image1.size)
print("processed_image shape:", processed_image1.shape)

# Define text descriptions
text_description1 = "A small blond boy wearing a green coat , blue pants and white shoes is standing close to and looking at a body of water ."
text_description2 = "A married man wearing a watch that appears to be about 7 o'clock is pouring cream into a coffee type beverage in a blue and white cup ."
text_description3 = "A black and white dog wearing a harness licks a brunette wearing glasses ."

text_descriptions = [text_description1, text_description2, text_description3]
texts = clip.tokenize(text_descriptions).to(device)
print("tokenized caption shape:",texts[0].shape)

# Image list for retrieval
images = [processed_image1, processed_image2, processed_image3]

# Query an image with text
# query_text = clip.tokenize(["a plot"]).to(device)

query_text = texts[0][np.newaxis,:]
print("tokenized query text shape:",query_text.shape)

image_probs, best_image_index = image_retrieval(images, query_text)

# Query a text with an image
# query_image = processed_image2
# text_probs, best_text_index = text_retrieval(query_image, texts)

# Display results
# print(f"Best matching image index: {best_image_index} -> {image_path1 if best_image_index == 0 else image_path2}")
# print(f"Best matching text index: {best_text_index} -> {text_descriptions[best_text_index]}")
# print("\nImage Similarity Scores:\n", image_probs)
# print("\nText Similarity Scores:\n", text_probs)
