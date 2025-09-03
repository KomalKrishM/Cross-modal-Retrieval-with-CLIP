import json
import os
from PIL import Image
import clip
import torch
import random


class CaptionExtractor:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path

    def extract_captions(self):
        with open(self.json_file_path, "r") as val: 
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

    def process_captions(self, redundant_captions):
        captions_dict = {}
        for k, v in redundant_captions.items():
            k = str(k).zfill(12)
            captions_dict[k] = v[0]
        return captions_dict

class ImageProcessor:
   def __init__(self, images_folder_path):
       self.images_folder_path = images_folder_path

   def load_images(self):
       images_dict = {}
       for image_no in os.listdir(self.images_folder_path):
           image_id = image_no.split(".")[0]
           images_dict[image_id] = os.path.join(self.images_folder_path, image_no)
       return images_dict

   @staticmethod
   def load_and_preprocess_image(image_path, device, preprocess):
       image = Image.open(image_path)
       processed_image = preprocess(image).unsqueeze(0).to(device)
       return image, processed_image

class DataSampler:
   @staticmethod
   def sample_data(captions_dict, images_dict, num_samples=500):
       sampled_captions = []
       sampled_caption_ids = []
       sampled_images_path = []
       sampled_image_ids = []
       for caption_id, caption in captions_dict.items():
           if random.random() > 0.5:
              if len(sampled_caption_ids) < num_samples:
                 sampled_captions.append(caption)
                 sampled_caption_ids.append(caption_id)
                 sampled_images_path.append(images_dict[caption_id])
                 sampled_image_ids.append(caption_id)
              else:
                 break
           else:
              continue
       return sampled_captions, sampled_caption_ids, sampled_images_path, sampled_image_ids
   
class CLIPRetriever:
   def __init__(self, model_name = "ViT-B/32"):
       self.device = 'mps' if torch.backends.mps.is_built() else "cpu"
       self.model, self.preprocess = clip.load(model_name, device = self.device)
    
   def retrieve_text(self, query_image, text_features):
       with torch.no_grad():
            image_features = self.model.encode_image(query_image)
       similarity = (image_features @ text_features.T).softmax(dim=-1)
       one_max_index = sorted(range(similarity.shape[1]), key = lambda i: similarity[0][i], reverse=True)[:1]
       five_max_indices = sorted(range(similarity.shape[1]), key = lambda i: similarity[0][i], reverse=True)[:5]
       ten_max_indices = sorted(range(similarity.shape[1]), key = lambda i: similarity[0][i], reverse=True)[:10]
       return similarity, one_max_index, five_max_indices, ten_max_indices

   def retrieve_image(self, query_text, images):
       with torch.no_grad():
           image_features = torch.cat(images)
           text_features = self.model.encode_text(query_text)
       similarity = (image_features @ text_features.T).softmax(dim=0)
       one_max_index = sorted(range(similarity.shape[0]), key = lambda i: similarity[i], reverse=True)[:1]
       five_max_indices = sorted(range(similarity.shape[0]), key = lambda i: similarity[i], reverse=True)[:5]
       ten_max_indices = sorted(range(similarity.shape[0]), key = lambda i: similarity[i], reverse=True)[:10]
       return similarity, one_max_index, five_max_indices, ten_max_indices
   
   def image_to_text_retrieval(self, images_path, captions, caption_ids, image_ids):
       tokenized_texts = clip.tokenize(captions).to(self.device)
       with torch.no_grad():
            text_features = self.model.encode_text(tokenized_texts)
       
       top_1_correctly_retrieved_texts = 0
       top_5_correctly_retrieved_texts = 0
       top_10_correctly_retrieved_texts = 0
       for image_id, image_path in zip(image_ids, images_path):
           _, preprocessed_img_query = ImageProcessor.load_and_preprocess_image(image_path, self.device, self.preprocess)
           similarity, top_one, top_five, top_ten = self.retrieve_text(preprocessed_img_query, text_features)
           
           top_one_cap_id = [caption_ids[i] for i in top_one]
           top_five_cap_ids = [caption_ids[i] for i in top_five]
           top_ten_cap_ids = [caption_ids[i] for i in top_ten]

           if image_id in top_one_cap_id:
               top_1_correctly_retrieved_texts += 1
           if image_id in top_five_cap_ids:
               top_5_correctly_retrieved_texts += 1
           if image_id in top_ten_cap_ids:
               top_10_correctly_retrieved_texts += 1

       return (100.0 * top_1_correctly_retrieved_texts/len(caption_ids), \
               100.0 * top_5_correctly_retrieved_texts/len(caption_ids), \
               100.0 * top_10_correctly_retrieved_texts/len(caption_ids))
   
   def text_to_image_retrieval(self, images_path, captions, caption_ids, image_ids):
       image_features = []
       for image_path in images_path:
           _, preprocessed_img = ImageProcessor.load_and_preprocess_image(image_path, self.device, self.preprocess)
           with torch.no_grad():
                image_feature = self.model.encode_image(preprocessed_img)
           image_features.append(image_feature)

       top_1_correctly_retrieved_images = 0
       top_5_correctly_retrieved_images = 0
       top_10_correctly_retrieved_images = 0      
       for caption_id, caption in zip(caption_ids, captions):
           tokenized_caption_query = clip.tokenize(caption).to(self.device)
           similarity, top_one, top_five, top_ten = self.retrieve_image(tokenized_caption_query, image_features)

           top_one_img_id = [image_ids[i] for i in top_one]
           top_five_img_ids = [image_ids[i] for i in top_five]
           top_ten_img_ids = [image_ids[i] for i in top_ten]

           if caption_id in top_one_img_id:
               top_1_correctly_retrieved_images += 1
           if caption_id in top_five_img_ids:
               top_5_correctly_retrieved_images += 1
           if caption_id in top_ten_img_ids:
               top_10_correctly_retrieved_images += 1

       return (100.0 * top_1_correctly_retrieved_images/len(caption_ids), \
               100.0 * top_5_correctly_retrieved_images/len(caption_ids), \
               100.0 * top_10_correctly_retrieved_images/len(caption_ids))
           
def main():

    json_file_path = "./annotations/captions_val2017.json"
    images_folder_path = "./val2017/"

    caption_extractor = CaptionExtractor(json_file_path)
    image_processor = ImageProcessor(images_folder_path)
    clip_retriever = CLIPRetriever()

    redundant_captions = caption_extractor.extract_captions()
    captions_dict = caption_extractor.process_captions(redundant_captions)
    images_dict = image_processor.load_images()

    n_trials = 10
    avg_i2t_top_1, avg_i2t_top_5, avg_i2t_top_10 = 0, 0, 0
    avg_t2i_top_1, avg_t2i_top_5, avg_t2i_top_10 = 0, 0, 0

    for trial_no in range(n_trials):
        captions, caption_ids, images_path, image_ids = DataSampler.sample_data(captions_dict, images_dict, num_samples=500)
        print(f"Trial {trial_no + 1}: Sampled {len(caption_ids)} captions and images for retrieval evaluation.")

        i2t_top_1, i2t_top_5, i2t_top_10 = clip_retriever.image_to_text_retrieval(images_path, captions, caption_ids, image_ids)
        print(f"Sampled Image to Text Retrieval Accuracies: top_1 {i2t_top_1} top_5 {i2t_top_5} top_10 {i2t_top_10}")

        t2i_top_1, t2i_top_5, t2i_top_10 = clip_retriever.text_to_image_retrieval(images_path, captions, caption_ids, image_ids)
        print(f"Sampled Text to Image Retrieval Accuracies: top_1 {t2i_top_1} top_5 {t2i_top_5} top_10 {t2i_top_10}")

        avg_i2t_top_1 += i2t_top_1
        avg_i2t_top_5 += i2t_top_5
        avg_i2t_top_10 += i2t_top_10

        avg_t2i_top_1 += t2i_top_1
        avg_t2i_top_5 += t2i_top_5
        avg_t2i_top_10 += t2i_top_10

    print(f"Average Image to Text Retrieval Accuracy: top_1 {avg_i2t_top_1 / n_trials} top_5 {avg_i2t_top_5 / n_trials} top_10 {avg_i2t_top_10 / n_trials}")
    print(f"Average Text to Image Retrieval Accuracy: top_1 {avg_t2i_top_1 / n_trials} top_5 {avg_t2i_top_5 / n_trials} top_10 {avg_t2i_top_10 / n_trials}")

if __name__ == "__main__":
    main()

