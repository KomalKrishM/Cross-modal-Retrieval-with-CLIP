import random
import shutil
import os
import torch
import clip
from PIL import Image
import numpy as np

source_caption_file_path = "./labels_flickr.txt"
source_image_file_path = "./flickr30k-images"
save_image_path = './Sampled_flickr30k-images/'
save_caption_path = './Sampled_flickr30k-captions/'

def sample_captions(source_caption_file_path, n_trail):
    captions_ids = []
    captions = []
    num_subsamples = 100

    with open(source_caption_file_path, 'r', encoding='utf-8', errors='ignore') as captions_file:
        for i, tot_caption in enumerate(captions_file):
            if i % 5 == 0:
                #  print(i, caption[:16])
                caption_id = tot_caption.split('#')[0]
                # id = caption_id.split('.')[0]
                captions_ids.append(caption_id)
                caption = tot_caption.split('\t')[1]
                captions.append(caption)

    captions_len = len(captions_ids)
    sampled_captions_ids = []
    sampled_captions = []
    sampled_ids = random.sample(range(0, captions_len - 1), num_subsamples)

    file_name = save_caption_path + f'test_captions_{n_trail}.txt'

    with open(file_name, 'w', encoding='utf-8', errors='ignore') as captions_file:
        for i in range(num_subsamples):
            sampled_captions_ids.append(captions_ids[sampled_ids[i]])
            sampled_captions.append(captions[sampled_ids[i]])
            captions_file.write(sampled_captions_ids[i] + '\t' + sampled_captions[i] + '\n')

    return sampled_captions, sampled_captions_ids


def sample_images(source_image_file_path, save_image_path, sampled_captions_ids, trail_no):

    if os.path.exists(save_image_path):
        shutil.rmtree(save_image_path)

    for caption_id in sampled_captions_ids:
        sampled_image = os.path.join(source_image_file_path, caption_id)
        save_path = os.path.join(save_image_path, caption_id)

        if not os.path.exists(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        shutil.copy2(sampled_image, save_path)

# Function to load and preprocess an image
def load_and_preprocess_image(image_path, device):
    image = Image.open(image_path)
    processed_image = preprocess(image).unsqueeze(0).to(device)
    return image, processed_image


# Function to compute similarity between a query image and text descriptions
def text_retrieval(query_image, text_features, model):
    with torch.no_grad():
        image_features = model.encode_image(query_image)

    similarity = (image_features @ text_features.T).softmax(dim=-1)

    one_max_index = sorted(range(similarity.shape[1]), key=lambda i: similarity[0][i], reverse=True)[:1]
    five_max_indices = sorted(range(similarity.shape[1]), key=lambda i: similarity[0][i], reverse=True)[:5]
    ten_max_indices = sorted(range(similarity.shape[1]), key=lambda i: similarity[0][i], reverse=True)[:10]

    return similarity, one_max_index, five_max_indices, ten_max_indices


# Function to compute similarity between images and a text query
def image_retrieval(images, query_text, model):
    with torch.no_grad():
        image_features = torch.cat(images)
        text_features = model.encode_text(query_text)
    similarity = (image_features @ text_features.T).softmax(dim=0)
    one_max_index = sorted(range(similarity.shape[0]), key=lambda i: similarity[i], reverse=True)[:1]
    five_max_indices = sorted(range(similarity.shape[0]), key=lambda i:similarity[i], reverse=True)[:5]
    ten_max_indices = sorted(range(similarity.shape[0]), key=lambda i:similarity[i], reverse=True)[:10]

    return similarity, one_max_index, five_max_indices, ten_max_indices


def image_to_text_retrieval(save_image_path, sampled_captions, sampled_captions_ids, model, device):
    # Process textual data
    tokenized_texts = clip.tokenize(sampled_captions).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokenized_texts)
    
    # image-to-text retrieval
    top_1_correctly_retrieved_texts = 0
    top_5_correctly_retrieved_texts = 0
    top_10_correctly_retrieved_texts = 0
    for image_id in os.listdir(save_image_path):
        
        image_path = os.path.join(save_image_path, image_id)
        _, preprocessed_img_query = load_and_preprocess_image(image_path, device)
        similarity, top_one, top_five, top_ten = text_retrieval(preprocessed_img_query, text_features, model)

        top_one_cap_id = [sampled_captions_ids[i] for i in top_one]
        top_five_cap_ids = [sampled_captions_ids[i] for i in top_five]
        top_ten_cap_ids = [sampled_captions_ids[i] for i in top_ten]

        if image_id in top_one_cap_id:
            top_1_correctly_retrieved_texts += 1
        if image_id in top_five_cap_ids:
            top_5_correctly_retrieved_texts += 1
        if image_id in top_ten_cap_ids:
            top_10_correctly_retrieved_texts += 1
    return 100.0 * (top_1_correctly_retrieved_texts / len(sampled_captions)), \
            100.0 * (top_5_correctly_retrieved_texts / len(sampled_captions)), \
            100.0 * (top_10_correctly_retrieved_texts / len(sampled_captions))


def text_to_image_retrieval(save_image_path, sampled_captions, sampled_captions_ids, model, device):
    # Process images
    image_features = []
    images_id = {}
    for i, image in enumerate(os.listdir(save_image_path)):
        images_id[i] = image
        image_path = os.path.join(save_image_path, image)
        img, preprocessed_img = load_and_preprocess_image(image_path, device)
        with torch.no_grad():
            image_feature = model.encode_image(preprocessed_img)
        image_features.append(image_feature)
    
    # text-to-image retrieval
    top_1_correctly_retrieved_images = 0
    top_5_correctly_retrieved_images = 0
    top_10_correctly_retrieved_images = 0
    for caption_id, caption in zip(sampled_captions_ids, sampled_captions):
     
        tokenized_caption_query = clip.tokenize(caption).to(device)
        similarity, top_one, top_five, top_ten = image_retrieval(image_features, tokenized_caption_query, model)
        
        top_one_img_id = [images_id[i] for i in top_one]
        top_five_img_ids = [images_id[i] for i in top_five]
        top_ten_img_ids = [images_id[i] for i in top_ten]
        if caption_id in top_one_img_id:
            top_1_correctly_retrieved_images += 1
        if caption_id in top_five_img_ids:
            top_5_correctly_retrieved_images += 1
        if caption_id in top_ten_img_ids:
            top_10_correctly_retrieved_images += 1

    return 100.0 * (top_1_correctly_retrieved_images / len(sampled_captions)), \
            100.0 * (top_5_correctly_retrieved_images / len(sampled_captions)), \
            100.0 * (top_10_correctly_retrieved_images / len(sampled_captions))


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

n_trails = 1
avg_i2t_top1, avg_i2t_top5, avg_i2t_top10 = 0, 0, 0
avg_t2i_top1, avg_t2i_top5, avg_t2i_top10 = 0, 0, 0

for trail_no in range(n_trails):
    sampled_captions, sampled_captions_ids = sample_captions(source_caption_file_path, trail_no)
    image_file_path = save_image_path + f'test_images_{trail_no}'
    sample_images(source_image_file_path, image_file_path, sampled_captions_ids, trail_no)

    i2t_top1, i2t_top5, i2t_top10 = image_to_text_retrieval(image_file_path, sampled_captions,
                                                                   sampled_captions_ids, model, device)
    print("Sampled Image to Text Retrieval Accuracies: top_1 {} top_5 {} top_10 {}".format(i2t_top1, i2t_top5, i2t_top10))

    t2i_top1, t2i_top5, t2i_top10 = text_to_image_retrieval(image_file_path, sampled_captions,
                                                                       sampled_captions_ids, model, device)
    print("Sampled Text to Image Retrieval Accuracies: top_1 {} top_5 {} top_10 {}".format(t2i_top1, t2i_top5, t2i_top10))

    avg_i2t_top1 += i2t_top1
    avg_i2t_top5 += i2t_top5
    avg_i2t_top10 += i2t_top10
    avg_t2i_top1 += t2i_top1
    avg_t2i_top5 += t2i_top5
    avg_t2i_top10 += t2i_top10

print(f"Average Image to Text Retrieval Accuracy: top_1 {avg_i2t_top1 / n_trails} top_5 {avg_i2t_top5 / n_trails} top_10 {avg_i2t_top10 / n_trails}")
print(f"Average Text to Image Retrieval Accuracy:top_1 {avg_t2i_top1 / n_trails} top_5 {avg_t2i_top5 / n_trails} top_10 {avg_t2i_top10 / n_trails}")

