import json
import os
from PIL import Image
import clip
import torch
import random
# from CMRwCLIP import load_and_preprocess_image, retrieve_image, retrieve_text

def extract_captions(val_json_file_path):

    with open(val_json_file_path, "r") as val:
        
        group_captions = {}
        data = json.load(val)
        annotations = data["annotations"]
        for item in annotations:
            image_id = item["image_id"]
            caption = item["caption"]
            if image_id in group_captions:
                group_captions[image_id].append(caption)
            else:
                group_captions[image_id] = [caption]
    return group_captions


def sample_data(captions_dict, images_dict, num_samples=100):
    sampled_captions = []
    sampled_caption_ids = []
    sampled_image_ids = []
    sampled_image_ids_path = []
    for caption_id, caption in captions_dict.items():
        
        if random.random() > 0.5:
            if len(sampled_caption_ids) < num_samples:
                sampled_caption_ids.append(caption_id)
                sampled_captions.append(caption)
                sampled_image_ids.append(caption_id)
                sampled_image_ids_path.append(images_dict[caption_id])
            else:
               break
        else:
            continue
    return sampled_caption_ids, sampled_captions, sampled_image_ids, sampled_image_ids_path

# Function to load and preprocess an image
def load_and_preprocess_image(image_path, device):
    image = Image.open(image_path)
    processed_image = preprocess(image).unsqueeze(0).to(device)
    return image, processed_image


# Function to compute similarity between a query image and text descriptions
def retrieve_text(query_image, text_features, model):
    with torch.no_grad():
        image_features = model.encode_image(query_image)
    #   text_features  = model.encode_text(texts)
    # print(image_features.shape)
    # pri
    similarity = (image_features @ text_features.T).softmax(dim=-1)
    # print(similarity.shape)

    # one_max_index_1 = similarity.argmax().item()
    one_max_index = sorted(range(similarity.shape[1]), key=lambda i: similarity[0][i], reverse=True)[:1]
    # print(one_max_index_1, one_max_index_2)
    five_max_indices = sorted(range(similarity.shape[1]), key=lambda i: similarity[0][i], reverse=True)[:5]
    ten_max_indices = sorted(range(similarity.shape[1]), key=lambda i: similarity[0][i], reverse=True)[:10]

    return similarity, one_max_index, five_max_indices, ten_max_indices


# Function to compute similarity between images and a text query
def retrieve_image(images, query_text, model):
    with torch.no_grad():
        image_features = torch.cat(images)
        text_features = model.encode_text(query_text)
        # print(image_features.shape)
        # print(text_features.shape)
        # pri
    similarity = (image_features @ text_features.T).softmax(dim=0)
    # print(similarity.shape)
    # one_max_index_1 = similarity.argmax().item()
    one_max_index = sorted(range(similarity.shape[0]), key=lambda i: similarity[i], reverse=True)[:1]
    # print(one_max_index_1, one_max_index_2)
    # pri
    five_max_indices = sorted(range(similarity.shape[0]), key=lambda i:similarity[i], reverse=True)[:5]
    ten_max_indices = sorted(range(similarity.shape[0]), key=lambda i:similarity[i], reverse=True)[:10]

    return similarity, one_max_index, five_max_indices, ten_max_indices

def image_to_text_retrieval(image_ids_path, captions, caption_ids, image_ids, model, device):
   # Process textual data
   tokenized_texts = clip.tokenize(captions).to(device)
   with torch.no_grad():
      text_features = model.encode_text(tokenized_texts)

   # image-to-text retrieval
   top_1_correctly_retrieved_texts = 0
   top_5_correctly_retrieved_texts = 0
   top_10_correctly_retrieved_texts = 0
   for image_id, image_path in zip(image_ids, image_ids_path):
      
      _, preprocessed_img_query = load_and_preprocess_image(image_path, device)
      similarity, top_one, top_five, top_ten= retrieve_text(preprocessed_img_query, text_features, model)

      top_one_cap_id = [caption_ids[i] for i in top_one]
      top_five_cap_ids = [caption_ids[i] for i in top_five]
      top_ten_cap_ids = [caption_ids[i] for i in top_ten]
    
      if image_id in top_one_cap_id:
        top_1_correctly_retrieved_texts += 1
      if image_id in top_five_cap_ids:
        top_5_correctly_retrieved_texts += 1
      if image_id in top_ten_cap_ids:
        top_10_correctly_retrieved_texts += 1

   return 100.0 * (top_1_correctly_retrieved_texts / len(caption_ids)), \
            100.0 * (top_5_correctly_retrieved_texts / len(caption_ids)), \
            100.0 * (top_10_correctly_retrieved_texts / len(caption_ids))

def text_to_image_retrieval(image_ids_path, captions, caption_ids, image_ids, model, device):
   # Process images
   image_features = []
   for image_path in image_ids_path:

      img, preprocessed_img = load_and_preprocess_image(image_path, device)
      with torch.no_grad():
         image_feature = model.encode_image(preprocessed_img)
      image_features.append(image_feature)

   # text-to-image retrieval
   top_1_correctly_retrieved_images = 0
   top_5_correctly_retrieved_images = 0
   top_10_correctly_retrieved_images = 0
   for caption_id, caption in zip(caption_ids, captions):

      tokenized_caption_query = clip.tokenize(caption).to(device)
      similarity, top_one, top_five, top_ten = retrieve_image(image_features, tokenized_caption_query, model)

      top_one_img_id = [image_ids[i] for i in top_one]
      top_five_img_ids = [image_ids[i] for i in top_five]
      top_ten_img_ids = [image_ids[i] for i in top_ten]
      if caption_id in top_one_img_id:
        top_1_correctly_retrieved_images += 1
      if caption_id in top_five_img_ids:
        top_5_correctly_retrieved_images += 1
      if caption_id in top_ten_img_ids:
        top_10_correctly_retrieved_images += 1

   return 100.0 * (top_1_correctly_retrieved_images / len(caption_ids)), \
            100.0 * (top_5_correctly_retrieved_images / len(caption_ids)), \
            100.0 * (top_10_correctly_retrieved_images / len(caption_ids))


redundant_captions_dict = extract_captions("/Users/komalkrishnamogilipalepu/Downloads/MSCOCO/MSCOCO_Val/annotations/captions_val2017.json")

# print(redundant_captions_dict[190236])  # Example to print captions for a specific image ID

# Uncomment below to write all captions to a single file with image IDs

# with open("ms_coco_val_captions.txt", "w") as out:
#     for image_id, captions in redundant_captions_dict.items():
#         for caption in captions:
#             out.write(f"{image_id}\t{caption}\n")
#         out.write("\n")

captions_dict = {k: v[0] for k, v in redundant_captions_dict.items()}  # Keep only the first caption for each image ID
# print(captions_dict[190236])  # Example to print the first caption for a specific image ID

captions_dict = {}
for k, v in redundant_captions_dict.items():
    k = str(k).zfill(12)  # Pad the image ID with leading zeros to make it 12 digits
    captions_dict[k] = v[0]  # Keep only the first caption for each image ID

device = "mps" if torch.backends.mps.is_built() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

images_folder_path = "/Users/komalkrishnamogilipalepu/Downloads/MSCOCO/MSCOCO_Val/val2017/"
images_dict = {}

for image_no in os.listdir(images_folder_path):
    image_id = image_no.split(".")[0]  # Extract the image ID without the file extension
    images_dict[image_id] = os.path.join(images_folder_path, image_no)
print(images_dict["000000093437"])  # Example to print the file path for a specific image ID

n_trails = 10
avg_i2t_top1, avg_i2t_top5, avg_i2t_top10 = 0, 0, 0
avg_t2i_top1, avg_t2i_top5, avg_t2i_top10 = 0, 0, 0

for trail_no in range(n_trails):
    caption_ids, captions, image_ids, image_ids_path = sample_data(captions_dict, images_dict, num_samples=100)
    print(f"Trail {trail_no + 1}: Sampled {len(caption_ids)} captions and images for retrieval evaluation.")
    i2t_top1, i2t_top5, i2t_top10 = image_to_text_retrieval(image_ids_path, captions,
                                                                   caption_ids, image_ids, clip_model, device)
    print("Sampled Image to Text Retrieval Accuracies: top_1 {} top_5 {} top_10 {}".format(i2t_top1, i2t_top5, i2t_top10))

    t2i_top1, t2i_top5, t2i_top10 = text_to_image_retrieval(image_ids_path, captions,
                                                                       caption_ids, image_ids, clip_model, device)
    print("Sampled Text to Image Retrieval Accuracies: top_1 {} top_5 {} top_10 {}".format(t2i_top1, t2i_top5, t2i_top10))

    avg_i2t_top1 += i2t_top1
    avg_i2t_top5 += i2t_top5
    avg_i2t_top10 += i2t_top10
    avg_t2i_top1 += t2i_top1
    avg_t2i_top5 += t2i_top5
    avg_t2i_top10 += t2i_top10

print(f"Average Image to Text Retrieval Accuracy: top_1 {avg_i2t_top1 / n_trails} top_5 {avg_i2t_top5 / n_trails} top_10 {avg_i2t_top10 / n_trails}")
print(f"Average Text to Image Retrieval Accuracy:top_1 {avg_t2i_top1 / n_trails} top_5 {avg_t2i_top5 / n_trails} top_10 {avg_t2i_top10 / n_trails}")
