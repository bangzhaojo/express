# Fluent but Unfeeling: The Emotional Blind Spots of Language Models
#### Bangzhao Shu*, Isha Joshi*, Melissa Karnaze, Anh C. Pham, Ishita Kakkar, Sindhu Kothe, Arpine Hovasapian, Mai ElSherief

## Abstract
The versatility of Large Language Models (LLMs) in natural language understanding has made them increasingly popular in mental health research. \textcolor{black}{While many studies explore LLMs' capabilities in emotion prediction, a critical gap remains in evaluating whether LLMs align with human emotions at a fine-grained level. Existing research typically focuses on classifying emotions into predefined, limited categories, overlooking more nuanced expressions. To address this gap, we introduce EXPRESS, a benchmark dataset curated from Reddit communities featuring 251 fine-grained, self-disclosed emotion labels. Our comprehensive evaluation framework examines predicted emotion terms and decomposes them into eight basic emotions using established emotion theories, enabling a fine-grained comparison. Systematic testing of prevalent LLMs under various prompt settings reveals that accurately predicting emotions that align with human self-disclosed emotions remains challenging. Qualitative analysis further shows that while certain LLMs generate emotion terms consistent with emotion theories and definitions, they often fail to capture contextual cues as effectively as human self-disclosures. These findings highlight the limitations of LLMs in fine-grained emotion alignment and offer insights for future research aimed at enhancing their contextual understanding.

## Code and Data
```                                
├── data/                       <- Project data
│   ├── human_evaluation/               <- Human evaluation results from the three coders
│   ├── emotion-lexicons.pkl            <- Comprehensive list of emotions used in our study
│   ├── express.csv                     <- EXPRESS dataset (original and full version)
│   ├── lexicon-decomposition.csv       <- Lexicons mapping words into eight basic emotions and sentiments
│
├── results/                    <- Results generated during experiments and analysis. Only sample (zero-shot) results are uploaded
│
├── src/                        <- Scripts to run experiments and analysis
│   ├── model-inference.py              <- Script to run inference of LLMs on EXPRESS
│   ├── construct-dataset/              <- Scripts to construct EXPRESS from raw Reddit data
│   │   ├── emotion-lexicon-masking.py  <- Script to mask emotions in raw Reddit text
│   │   ├── post-segmentation.py        <- Script to segment long posts into chunks of 512 tokens
│   │
│   ├── evaluation/                     <- Scripts for result evaluation
│       ├── result-cleaning.py          <- Script to format model results
│       ├── result-evaluation.py        <- Script to evaluate model results
│
└── README.md
```
