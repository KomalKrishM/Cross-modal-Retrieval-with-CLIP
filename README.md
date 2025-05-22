This work implements the cross-modal retrieval using Contrastive Language Image Pretraining model (CLIP) on Flickr30K dataset. Given a query image CLIP model retrives the most suitable caption and viceversa. 


The file CRMwCLIP.py generates a test set of 1000 samples from Flick30K and provide the retrieval rate for recall values 1,5, and 10.
The file CLIP_Analysis.py gives the demo for finding the best matched caption/image for the queried image/caption.

Recall@1 - indicates that the percentage of ground truth that match to the retrieved top-1 image/caption.
Recall@5 - indicates that the percentage of ground truth lies in the retrieved top-5 images/captions.
Recall@10 - indicates that the percentage of ground truth lies in the retrieved top-10 images/captions.


Results on image to text retrieval: top-1 accuracy/Recall@1 is 30.22, top-5 accuracy/Recall@5 is 62.88, and top-10 accuracy/Recall@10 is 75.88
Results on text to image retrieval: top-1 accuracy/Recall@1 is 61.06, top-5 accuracy/Recall@5 is 85.06, and top-10 accuracy/Recall@10 is 90.86

