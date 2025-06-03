

# Cross-Modal Retrieval with CLIP on Flickr30K

This project implements **cross-modal retrieval** using the **Contrastive Language–Image Pretraining (CLIP)** model on the **Flickr30K** dataset.

The system can:
- Retrieve the most relevant **caption** for a given **image**
- Retrieve the most relevant **image** for a given **caption**

### Example: Image-to-Text Retrieval

**Query (Image):** 
![Example Query Image](6734417.jpg)

**Captions:**
1. "A small blond boy wearing a green coat , blue pants and white shoes is standing close to and looking at a body of water ."   *Similarity - 0.79*
2. "A kid is playing at the seashore with his friends."   *Similarity - 0.22*
3. "A Kid is eating food while watching a movie"   *Similarity - 0.00006*

---

## Files

- `CRMwCLIP.py`  
  Generates a test set of 1,000 samples from Flickr30K and evaluates retrieval performance using Recall@1, Recall@5, and Recall@10.

- `CLIP_Analysis.py`  
  Demo script for retrieving the best-matching caption for a queried image, or vice versa.

---

## Metrics Explained

**Recall@K** indicates whether the correct match appears in the top K results returned by the model:

- **Recall@1**: Ground truth is the top-1 result
- **Recall@5**: Ground truth is in the top-5 results
- **Recall@10**: Ground truth is in the top-10 results

---

## Results

### Image-to-Text Retrieval

| Metric     | Value (%) |
|------------|------------|
| Recall@1   | 30.22      |
| Recall@5   | 62.88      |
| Recall@10  | 75.88      |

### Text-to-Image Retrieval

| Metric     | Value (%) |
|------------|------------|
| Recall@1   | 61.06      |
| Recall@5   | 85.06      |
| Recall@10  | 90.86      |

---

## Requirements

- Python 3.x
- PyTorch
- OpenAI's CLIP model
- Flickr30K dataset

---

## Usage

```bash
# Evaluate retrieval performance
python CRMwCLIP.py

# Run demo analysis
python CLIP_Analysis.py

```

---

## Acknowledgement

If you find this work is useful, appreciate your acknowledgement to this repository.  


## References 

This project uses the [Flickr30K dataset](http://shannon.cs.illinois.edu/DenotationGraph/) for training and evaluation.

@article{young2014image,
  title={From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions},
  author={Young, Peter and Lai, Alice and Hodosh, Micah and Hockenmaier, Julia},
  journal={Transactions of the association for computational linguistics},
  volume={2},
  pages={67--78},
  year={2014},
  publisher={MIT Press One Rogers Street, Cambridge, MA 02142-1209, USA journals-info~…}
}

This project uses the [CLIP model](https://github.com/openai/CLIP)

@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle={International conference on machine learning},
  pages={8748--8763},
  year={2021},
  organization={PmLR}
}

Python Software Foundation. Python Language Reference, version 3.X.
Available at: https://www.python.org/

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G.,
Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A.,
Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S.,
Steiner, B., Fang, L., Bai, J., & Chintala, S.
PyTorch: An Imperative Style, High-Performance Deep Learning Library.
In NeurIPS 2019.

I used ChatGPT for refactoring the code
OpenAI. (2024). ChatGPT (May 22 version) [Large language model]. https://chat.openai.com

