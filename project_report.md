# AI Agent for Resume Screening and Job Recommendation using Resume Parsing

## Abstract
This project presents an end-to-end AI-powered recruitment assistant that automates resume screening
and job recommendation. The system leverages transformer-based language models (BERT, RoBERTa) for
resume classification and a Sentence-BERT semantic similarity engine for job recommendation, wrapped
in an interactive Streamlit web application deployed via ngrok from Google Colab.

## Objective
- Automate resume screening by classifying resumes into job categories.
- Recommend the most relevant job roles for a given resume using semantic similarity.
- Parse structured information (skills, education, experience) from unstructured resume text.
- Provide a professional, deployable web interface for real-time use.

## Methodology
1. Dataset acquisition from Kaggle (`gauravduttakiit/resume-dataset`).
2. Text cleaning and preprocessing (lowercasing, URL/special character removal, whitespace normalization).
3. Label encoding of job categories.
4. Fine-tuning BERT (`bert-base-uncased`) and RoBERTa (`roberta-base`) for multi-class classification.
5. Building a Sentence-BERT (`all-MiniLM-L6-v2`) based cosine-similarity recommendation engine over a
   20-role job description catalog.
6. Rule-based resume parsing (skills, education, experience) from PDF/DOCX/TXT uploads.
7. Deployment as a Streamlit web application exposed publicly via ngrok.

## Model Architecture
- **BERT**: 12-layer bidirectional transformer encoder, fine-tuned with a classification head
  (`BertForSequenceClassification`) over 25 job categories.
- **RoBERTa**: Robustly optimized BERT pretraining approach, fine-tuned identically to BERT for
  direct comparison.
- **Sentence-BERT**: Pretrained `all-MiniLM-L6-v2` sentence embedding model used (without fine-tuning)
  to compute semantic similarity between resumes and job descriptions.

## Preprocessing Steps
- Lowercasing all resume text.
- Removing URLs and email addresses.
- Removing special characters and non-alphanumeric symbols.
- Normalizing whitespace.
- Removing duplicate resumes.
- Dropping near-empty resume records (< 20 characters).

## Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score (per-class classification report)
- Confusion matrix

## Results Table
| Model | Accuracy |
|-------|----------|
| BERT (bert-base-uncased) | 90.16% |
| RoBERTa (roberta-base) | 99.48% |

## Conclusion
The system successfully combines discriminative transformer classifiers with a semantic similarity
recommendation engine to deliver an automated, explainable resume screening pipeline. The Streamlit
interface makes the pipeline accessible as an interactive recruitment tool suitable for demonstration
and prototyping purposes.

## Future Scope
- Expand the job description catalog with real-world job postings via live APIs.
- Fine-tune Sentence-BERT on domain-specific resume-job pairs for improved recommendation accuracy.
- Add named entity recognition (NER) for more robust skill/education/experience extraction.
- Deploy on a persistent cloud server (AWS/GCP/Azure) instead of ngrok for production use.
- Add multi-language resume support.
