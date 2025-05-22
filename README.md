This work implements the cross-modal retrieval using Contrastive Language Image Pretraining model (CLIP) on Flickr30K dataset. Given a query image CLIP model retrives the most suitable caption and viceversa. 


The file CRMwCLIP.py generates a test set of 1000 samples from Flick30K and provide the retrieval rate for recall values 1,5, and 10.
The file CLIP_Analysis.py gives the demo for finding the best matched caption/image for the queried image/caption.

Recall@1 - indicates that the percentage of ground truth that match to the retrieved top-1 image/caption.
Recall@5 - indicates that the percentage of ground truth lies in the retrieved top-5 images/captions.
Recall@10 - indicates that the percentage of ground truth lies in the retrieved top-10 images/captions.


Results on image to text retrieval: top-1 accuracy/Recall@1 is 30.22, top-5 accuracy/Recall@5 is 62.88, and top-10 accuracy/Recall@10 is 75.88
Results on text to image retrieval: top-1 accuracy/Recall@1 is 61.06, top-5 accuracy/Recall@5 is 85.06, and top-10 accuracy/Recall@10 is 90.86


# 📘 Cross-Modal Retrieval with CLIP on Flickr30K

This project implements **cross-modal retrieval** using the **Contrastive Language–Image Pretraining (CLIP)** model on the **Flickr30K** dataset.

The system can:
- Retrieve the most relevant **caption** for a given **image**
- Retrieve the most relevant **image** for a given **caption**

---

## 📂 Files

- `CRMwCLIP.py`  
  Generates a test set of 1,000 samples from Flickr30K and evaluates retrieval performance using Recall@1, Recall@5, and Recall@10.

- `CLIP_Analysis.py`  
  Demo script for retrieving the best-matching caption for a queried image, or vice versa.

---

## 📊 Metrics Explained

**Recall@K** indicates whether the correct match appears in the top K results returned by the model:

- **Recall@1**: Ground truth is the top-1 result
- **Recall@5**: Ground truth is in the top-5 results
- **Recall@10**: Ground truth is in the top-10 results

---

## 📈 Results

### 🔁 Image-to-Text Retrieval

| Metric     | Value (%) |
|------------|------------|
| Recall@1   | 30.22      |
| Recall@5   | 62.88      |
| Recall@10  | 75.88      |

### 🔁 Text-to-Image Retrieval

| Metric     | Value (%) |
|------------|------------|
| Recall@1   | 61.06      |
| Recall@5   | 85.06      |
| Recall@10  | 90.86      |

---

## 🛠 Requirements

- Python 3.x
- PyTorch
- OpenAI's CLIP model
- Flickr30K dataset

---

## 🧪 Usage

```bash
# Evaluate retrieval performance
python CRMwCLIP.py

# Run demo analysis
python CLIP_Analysis.py


