# Multiclass Mental Health Sentiment Analysis & Safety Routing System

> [!CAUTION]
> **IMPORTANT MEDICAL DISCLAIMER:** This software application is an experimental machine learning classification model developed strictly for academic evaluation, pattern identification, and data analysis research. It is **NOT** a diagnostic tool, a substitute for professional clinical judgment, or a medical intervention system. If you or someone you know is struggling with mental health challenges, emotional distress, or experiencing a crisis, please seek immediate guidance from qualified healthcare providers, psychiatrists, or professional crisis intervention helplines.

---

## 🎯 Project Overview

This project builds a multiclass NLP classifier using social media posts to identify underlying sentiment flags. Recognizing that standard engineering metrics (such as raw accuracy) are insufficient for high-risk applications, this architecture implements a customized **Safety Prioritization Logic Block** directly onto the model's prediction probability array ($P(y|x)$).

### Core Features:
- **7-Class Taxonomy Modeling:** Categorizes unstructured inputs across `Normal`, `Anxiety`, `Bipolar`, `Depression`, `Personality disorder`, `Stress`, and `Suicidal` markers.
- **Custom NLP Preprocessing Loop:** Removes systematic noise (HTML tags, URLs, mathematical anomalies, structural punctuation) without altering critical text tokens.
- **Sublinear TF-IDF Vectorization:** Smooths extreme word frequency spikes via sublinear transformation scales ($\log(1 + \text{tf})$).
- **Safety Threshold Triage System:** Computes custom fallback levels, escalating ambiguous or critical predictions into priority queues based on model confidence intervals.

---

## 📊 Dataset Evaluation & Characterization

The system is trained on a corpus of **53,043 source samples** tracking conversational entries.

### Imbalance Profiles & Baseline Densities:

| Target Class Label | Total Volumetric Yield | Normalized Sample Density (%) | Assigned Numeric Map ID |
| :--- | :---: | :---: | :---: |
| **Normal** | 16,351 | 30.83% | 3 |
| **Depression** | 15,404 | 29.04% | 2 |
| **Suicidal** | 10,653 | 20.08% | 6 |
| **Anxiety** | 3,888 | 7.33% | 0 |
| **Bipolar** | 2,877 | 5.42% | 1 |
| **Stress** | 2,669 | 5.03% | 5 |
| **Personality Disorder** | 1,201 | 2.26% | 4 |

### Structural Sequence Distributions:
- **Corpus Token Mean ($\mu$):** 113.2 words per sample string.
- **Standard Deviation ($\sigma$):** 163.7 words (high variance).
- **Upper Sequence Extremum (Max):** 6,300 terms.
- **Missing Value Handling:** 362 null or empty conversational strings were isolated and dropped during data cleaning to preserve structural integrity.

---

## 🏗️ Technical Pipeline & Feature Engineering

### 1. Advanced Text Preprocessing
Text fields pass through a multi-stage cleaning regex cascade:
- Case normalization (lowercase adjustment).
- Stripping of implicit protocol identifiers (`http/https`) and active domain schemas (`www`).
- Complete filtering of raw markup nodes (`<.*?>`), integer sets, and structural character arrays (`string.punctuation`).
- String volume filtering: Any entries yielding $\le 2$ total characters post-cleanup are automatically excluded to eliminate void input arrays.

### 2. Feature Extraction Configuration
Optimized unigram and bigram strings are built using an automated `TfidfVectorizer` mapping a **15,000 max feature vocabulary**:
- **Stop Word Exclusion:** English base lexicon.
- **Minimum Document Frequency ($\text{min\_df} = 3$):** Drops rare tokens or typing anomalies.
- **Sublinear Scaling:** Applies a logarithmic scaling function to raw term counts:

$$\text{tf}_{\text{scaled}} = 1 + \log(\text{tf})$$

This limits the skewing impact of highly repetitive words within single lengthy posts.

---

## ⚠️ The Safety Prioritization Framework

To bridge the gap between static machine learning classifications and real-world deployment safety, predictions are routed through an algorithmic escalation filter:

```text
    [ Raw Class Probabilities P(y|x) ]
                   │
                   ▼
      Is Max Predicted Label = 'Suicidal'? 
         ├── YES ──> Is Confidence >= 70%? ──> YES ──> [ CRITICAL ]
         │                                  └── NO  ──> [ HIGH RISK ]
         └── NO  ───> Is Label = 'Anxiety' AND Confidence >= 70%?
                        ├── YES ──> [ HIGH RISK ]
                        └── NO  ──> Is Max Confidence < 50%?
                                       ├── YES ──> [ NEEDS HUMAN REVIEW ]
                                       └── NO  ──> [ ROUTINE ]
