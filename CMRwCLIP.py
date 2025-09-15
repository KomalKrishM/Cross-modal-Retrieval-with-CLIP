import random
import shutil
import os
import torch
import clip
from PIL import Image


class DataSampler:
    def __init__(self, source_caption_file_path, source_image_file_path, save_image_path, save_caption_path):
        self.source_caption_file_path = source_caption_file_path
        self.source_image_file_path = source_image_file_path
        self.save_image_path = save_image_path
        self.save_caption_path = save_caption_path
        self.num_samples =500

    def sample_captions(self, n_trail):
        captions_ids = []
        captions = []

        with open(self.source_caption_file_path, 'r', encoding='utf-8', errors='ignore') as captions_file:
            for i, caption_info in enumerate(captions_file):
                if i % 5 == 0:
                    caption_id = caption_info.split('#')[0]
                    captions_ids.append(caption_id)
                    caption = caption_info.split('\t')[1]
                    captions.append(caption)

        captions_len = len(captions_ids)
        sampled_captions_ids = []
        sampled_captions = []
        sampled_inds = random.sample(range(0, captions_len - 1), self.num_samples)

        file_name = os.path.join(self.save_caption_path, f'test_captions_{n_trail}.txt')
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        with open(file_name, 'w', encoding='utf-8', errors='ignore') as captions_file:
            for i in range(len(sampled_inds)):
                sampled_captions_ids.append(captions_ids[sampled_inds[i]])
                sampled_captions.append(captions[sampled_inds[i]])
                captions_file.write(sampled_captions_ids[i] + '\t' + sampled_captions[i] + '\n')

        return sampled_captions, sampled_captions_ids

    def sample_images(self, sampled_captions_ids, trail_no):
        image_file_path = os.path.join(self.save_image_path, f'test_images_{trail_no}')
        if os.path.exists(image_file_path):
            shutil.rmtree(image_file_path)

        for caption_id in sampled_captions_ids:
            sampled_image = os.path.join(self.source_image_file_path, caption_id)
            save_path = os.path.join(image_file_path, caption_id)

            if not os.path.exists(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            shutil.copy2(sampled_image, save_path)

        return image_file_path
    

class CLIPRetrieval:
    def __init__(self, model_name="ViT-B/32"):
        self.device = "mps" if torch.backends.mps.is_built() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def load_and_preprocess_image(self, image_path):
        image = Image.open(image_path)
        processed_image = self.preprocess(image).unsqueeze(0).to(self.device)
        return image, processed_image

    def retrieve_text(self, query_image, text_features):
        with torch.no_grad():
            image_features = self.model.encode_image(query_image)

        similarity = (image_features @ text_features.T).softmax(dim=-1)
        one_max_index = sorted(range(similarity.shape[1]), key=lambda i: similarity[0][i], reverse=True)[:1]
        five_max_indices = sorted(range(similarity.shape[1]), key=lambda i: similarity[0][i], reverse=True)[:5]
        ten_max_indices = sorted(range(similarity.shape[1]), key=lambda i: similarity[0][i], reverse=True)[:10]

        return similarity, one_max_index, five_max_indices, ten_max_indices

    def retrieve_image(self, image_features, query_text):
        with torch.no_grad():
            image_features = torch.cat(image_features)
            text_features = self.model.encode_text(query_text)
        similarity = (image_features @ text_features.T).softmax(dim=0)
        one_max_index = sorted(range(similarity.shape[0]), key=lambda i: similarity[i], reverse=True)[:1]
        five_max_indices = sorted(range(similarity.shape[0]), key=lambda i: similarity[i], reverse=True)[:5]
        ten_max_indices = sorted(range(similarity.shape[0]), key=lambda i: similarity[i], reverse=True)[:10]

        return similarity, one_max_index, five_max_indices, ten_max_indices

    def image_to_text_retrieval(self, image_file_path, sampled_captions, sampled_captions_ids):
        tokenized_texts = clip.tokenize(sampled_captions).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokenized_texts)

        top_1_correctly_retrieved_texts = 0
        top_5_correctly_retrieved_texts = 0
        top_10_correctly_retrieved_texts = 0
        for image_id in os.listdir(image_file_path):
            image_path = os.path.join(image_file_path, image_id)
            _, preprocessed_img_query = self.load_and_preprocess_image(image_path)
            similarity, top_one, top_five, top_ten = self.retrieve_text(preprocessed_img_query, text_features)

            top_one_cap_id = [sampled_captions_ids[i] for i in top_one]
            top_five_cap_ids = [sampled_captions_ids[i] for i in top_five]
            top_ten_cap_ids = [sampled_captions_ids[i] for i in top_ten]

            if image_id in top_one_cap_id:
                top_1_correctly_retrieved_texts += 1
            if image_id in top_five_cap_ids:
                top_5_correctly_retrieved_texts += 1
            if image_id in top_ten_cap_ids:
                top_10_correctly_retrieved_texts += 1

        return (100.0 * top_1_correctly_retrieved_texts / len(sampled_captions),
                100.0 * top_5_correctly_retrieved_texts / len(sampled_captions),
                100.0 * top_10_correctly_retrieved_texts / len(sampled_captions))

    def text_to_image_retrieval(self, image_file_path, sampled_captions, sampled_captions_ids):
        image_features = []
        images_id = {}
        for i, image in enumerate(os.listdir(image_file_path)):
            images_id[i] = image
            image_path = os.path.join(image_file_path, image)
            _, preprocessed_img = self.load_and_preprocess_image(image_path)
            with torch.no_grad():
                image_feature = self.model.encode_image(preprocessed_img)
            image_features.append(image_feature)

        top_1_correctly_retrieved_images = 0
        top_5_correctly_retrieved_images = 0
        top_10_correctly_retrieved_images = 0
        for caption_id, caption in zip(sampled_captions_ids, sampled_captions):
            tokenized_caption_query = clip.tokenize(caption).to(self.device)
            similarity, top_one, top_five, top_ten = self.retrieve_image(image_features, tokenized_caption_query)

            top_one_img_id = [images_id[i] for i in top_one]
            top_five_img_ids = [images_id[i] for i in top_five]
            top_ten_img_ids = [images_id[i] for i in top_ten]

            if caption_id in top_one_img_id:
                top_1_correctly_retrieved_images += 1
            if caption_id in top_five_img_ids:
                top_5_correctly_retrieved_images += 1
            if caption_id in top_ten_img_ids:
                top_10_correctly_retrieved_images += 1

        return (100.0 * top_1_correctly_retrieved_images / len(sampled_captions),
                100.0 * top_5_correctly_retrieved_images / len(sampled_captions),
                100.0 * top_10_correctly_retrieved_images / len(sampled_captions))

def run(source_caption_file_path, source_image_file_path, save_image_path, save_caption_path, n_trails = 1):
    avg_i2t_top1, avg_i2t_top5, avg_i2t_top10 = 0, 0, 0
    avg_t2i_top1, avg_t2i_top5, avg_t2i_top10 = 0, 0, 0

    clip_retrieval = CLIPRetrieval()
    data_sampler = DataSampler(source_caption_file_path, source_image_file_path, save_image_path, save_caption_path)

    for trail_no in range(n_trails):
        sampled_captions, sampled_captions_ids = data_sampler.sample_captions(trail_no)
        image_file_path = data_sampler.sample_images(sampled_captions_ids, trail_no)

        i2t_top1, i2t_top5, i2t_top10 = clip_retrieval.image_to_text_retrieval(image_file_path, sampled_captions, sampled_captions_ids)
        print(f"Trail_{trail_no} Sampled Image to Text Retrieval Accuracies: top_1 {i2t_top1} top_5 {i2t_top5} top_10 {i2t_top10}")

        t2i_top1, t2i_top5, t2i_top10 = clip_retrieval.text_to_image_retrieval(image_file_path, sampled_captions, sampled_captions_ids)
        print(f"Trail_{trail_no} Sampled Text to Image Retrieval Accuracies: top_1 {t2i_top1} top_5 {t2i_top5} top_10 {t2i_top10}")

        avg_i2t_top1 += i2t_top1
        avg_i2t_top5 += i2t_top5
        avg_i2t_top10 += i2t_top10
        avg_t2i_top1 += t2i_top1
        avg_t2i_top5 += t2i_top5
        avg_t2i_top10 += t2i_top10

    print(f"Average Image to Text Retrieval Accuracy: top_1 {avg_i2t_top1 / n_trails} top_5 {avg_i2t_top5 / n_trails} top_10 {avg_i2t_top10 / n_trails}")
    print(f"Average Text to Image Retrieval Accuracy: top_1 {avg_t2i_top1 / n_trails} top_5 {avg_t2i_top5 / n_trails} top_10 {avg_t2i_top10 / n_trails}")

if __name__ == "__main__":
    source_caption_file_path = "./labels_flickr.txt"
    source_image_file_path = "/Users/komalkrishnamogilipalepu/Downloads/Cross_modal_Retrieval/flickr30k-images"
    save_image_path = './Sampled_flickr30k-images/'
    save_caption_path = './Sampled_flickr30k-captions/'

    run(source_caption_file_path, source_image_file_path, save_image_path, save_caption_path,n_trails=5)
 
