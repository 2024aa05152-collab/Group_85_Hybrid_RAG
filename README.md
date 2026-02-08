# Group 85: Hybrid RAG System with Automated Evaluation

## 📖 Overview
A hybrid Retrieval-Augmented Generation system combining dense vector retrieval (sentence transformers), sparse keyword retrieval (BM25), and Reciprocal Rank Fusion (RRF) to answer questions from 500 Wikipedia articles.

### Key Components:

- **Hybrid Retrieval**: Integrates **FAISS** (Dense) and **BM25** (Sparse) via **Reciprocal Rank Fusion (RRF)**.
- **Local Generation**: Uses **Qwen-2.5-0.5B-Instruct** for private, local-first inference.
- **Automated Benchmarking**: A comprehensive evaluation suite measuring MRR, Hit Rate, and Lexical Overlap.
- **Automated Evaluation**: 100 generated questions with MRR and custom metrics
- **Interactive UI**: Streamlit dashboard
- **One-command Pipeline**: Full automation from indexing to evaluation

 ### Important links:
 1) GitHub Repository: https://github.com/2024aa05152-collab/Group_85_Hybrid_RAG
 2) Fixed 200 Wikipedia URLs: https://github.com/2024aa05152-collab/Group_85_Hybrid_RAG/blob/develop/data/fixed_urls.json
 3) Random 300 Wikipedia URLs: https://github.com/2024aa05152-collab/Group_85_Hybrid_RAG/blob/develop/data/random_urls.json
 4) 100 Questions generated from 500 Wikipedia URLs: https://github.com/2024aa05152-collab/Group_85_Hybrid_RAG/blob/develop/data/questions_100.json
 5) Corpus chunks: https://raw.githubusercontent.com/2024aa05152-collab/Group_85_Hybrid_RAG/refs/heads/develop/data/corpus_chunks.json
 6) Evaluation results: https://github.com/2024aa05152-collab/Group_85_Hybrid_RAG/tree/develop/outputs
 7) dense.index: https://github.com/2024aa05152-collab/Group_85_Hybrid_RAG/blob/develop/data/dense.index
 8) sparse.pkl: https://github.com/2024aa05152-collab/Group_85_Hybrid_RAG/blob/develop/data/sparse.pkl
 9) consolidated evaluation results: https://raw.githubusercontent.com/2024aa05152-collab/Group_85_Hybrid_RAG/refs/heads/develop/data/evaluation_results.json

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python 3.9+
- 8GB RAM (Minimum) | 16GB (Recommended)
- ~2GB storage for model weights and indices
- [Optional] CUDA-enabled GPU for faster inference.

### 2. Environment Setup instructions
1) git clone -b develop https://github.com/2024aa05152-collab/Group_85_Hybrid_RAG.git
2) Open a terminal and create virtual environment python -m venv venv 
3) Activate it:
     Windows: .\venv\Scripts\activate 
     Mac/Linux: source venv/bin/activate
4) pip install -r requirements.txt
5) Create a .env file and place in project root in which provide the Hugging face token under HF_TOKEN
6) streamlit run app/streamlit_app.py

## requirements
https://github.com/2024aa05152-collab/Group_85_Hybrid_RAG/blob/develop/requirements.txt

## 📊 System Architecture
1) Embedding Model: all-MiniLM-L6-v2 (Dense)
2) Keyword Engine: BM25Okapi (Sparse)
3) Fusion: Reciprocal Rank Fusion (RRF) with k=60
4) LLM: Qwen2.5-0.5B-Instruct (Local Inference)

---

### Fixed 200 Wikipedia URLs in JSON format
{
  "science_and_technology": [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/History_of_artificial_intelligence",
    "https://en.wikipedia.org/wiki/AI_alignment",
    "https://en.wikipedia.org/wiki/Explainable_artificial_intelligence",
    "https://en.wikipedia.org/wiki/Artificial_general_intelligence",
    "https://en.wikipedia.org/wiki/Expert_system",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Supervised_learning",
    "https://en.wikipedia.org/wiki/Unsupervised_learning",
    "https://en.wikipedia.org/wiki/Semi-supervised_learning",
    "https://en.wikipedia.org/wiki/Reinforcement_learning",
    "https://en.wikipedia.org/wiki/Statistical_learning_theory",
    "https://en.wikipedia.org/wiki/Artificial_neural_network",
    "https://en.wikipedia.org/wiki/Deep_learning",
    "https://en.wikipedia.org/wiki/Convolutional_neural_network",
    "https://en.wikipedia.org/wiki/Recurrent_neural_network",
    "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)",
    "https://en.wikipedia.org/wiki/Backpropagation",
    "https://en.wikipedia.org/wiki/Natural_language_processing",
    "https://en.wikipedia.org/wiki/Language_model",
    "https://en.wikipedia.org/wiki/Large_language_model",
    "https://en.wikipedia.org/wiki/Word_embedding",
    "https://en.wikipedia.org/wiki/Text_mining",
    "https://en.wikipedia.org/wiki/Information_extraction",
    "https://en.wikipedia.org/wiki/Computer_vision",
    "https://en.wikipedia.org/wiki/Image_processing",
    "https://en.wikipedia.org/wiki/Object_detection",
    "https://en.wikipedia.org/wiki/Facial_recognition_system",
    "https://en.wikipedia.org/wiki/Data_science",
    "https://en.wikipedia.org/wiki/Big_data",
    "https://en.wikipedia.org/wiki/Data_mining",
    "https://en.wikipedia.org/wiki/Exploratory_data_analysis",
    "https://en.wikipedia.org/wiki/Feature_engineering",
    "https://en.wikipedia.org/wiki/Model_selection",
    "https://en.wikipedia.org/wiki/Cross-validation_(statistics)",
    "https://en.wikipedia.org/wiki/Evaluation_of_machine_learning_algorithms",
    "https://en.wikipedia.org/wiki/MLOps",
    "https://en.wikipedia.org/wiki/AutoML",
    "https://en.wikipedia.org/wiki/Algorithmic_bias",
    "https://en.wikipedia.org/wiki/Machine_learning_bias"
  ],

  "history_politics": [
    "https://en.wikipedia.org/wiki/History",
    "https://en.wikipedia.org/wiki/Political_history",
    "https://en.wikipedia.org/wiki/World_history",
    "https://en.wikipedia.org/wiki/Ancient_history",
    "https://en.wikipedia.org/wiki/Modern_history",
    "https://en.wikipedia.org/wiki/History_of_Europe",
    "https://en.wikipedia.org/wiki/History_of_Asia",
    "https://en.wikipedia.org/wiki/History_of_Africa",
    "https://en.wikipedia.org/wiki/History_of_the_United_States",
    "https://en.wikipedia.org/wiki/History_of_India",
    "https://en.wikipedia.org/wiki/Timeline_of_Earth",
    "https://en.wikipedia.org/wiki/Historiography",
    "https://en.wikipedia.org/wiki/Political_science",
    "https://en.wikipedia.org/wiki/Comparative_politics",
    "https://en.wikipedia.org/wiki/International_relations",
    "https://en.wikipedia.org/wiki/Government",
    "https://en.wikipedia.org/wiki/State_(polity)",
    "https://en.wikipedia.org/wiki/Constitution",
    "https://en.wikipedia.org/wiki/Democracy",
    "https://en.wikipedia.org/wiki/Republic",
    "https://en.wikipedia.org/wiki/Monarchy",
    "https://en.wikipedia.org/wiki/Communism",
    "https://en.wikipedia.org/wiki/Socialism",
    "https://en.wikipedia.org/wiki/Capitalism",
    "https://en.wikipedia.org/wiki/Fascism",
    "https://en.wikipedia.org/wiki/Cold_War",
    "https://en.wikipedia.org/wiki/World_War_I",
    "https://en.wikipedia.org/wiki/World_War_II",
    "https://en.wikipedia.org/wiki/French_Revolution",
    "https://en.wikipedia.org/wiki/American_Revolution",
    "https://en.wikipedia.org/wiki/Russian_Revolution",
    "https://en.wikipedia.org/wiki/United_Nations",
    "https://en.wikipedia.org/wiki/European_Union",
    "https://en.wikipedia.org/wiki/Non-Aligned_Movement",
    "https://en.wikipedia.org/wiki/Indian_National_Congress",
    "https://en.wikipedia.org/wiki/Organisation_of_Islamic_Cooperation",
    "https://en.wikipedia.org/wiki/Mahatma_Gandhi",
    "https://en.wikipedia.org/wiki/Winston_Churchill",
    "https://en.wikipedia.org/wiki/Nelson_Mandela",
    "https://en.wikipedia.org/wiki/Margaret_Thatcher"
  ],

  "geography_environment": [
    "https://en.wikipedia.org/wiki/Global_North_and_Global_South",
    "https://en.wikipedia.org/wiki/Least_developed_countries",
    "https://en.wikipedia.org/wiki/Landlocked_country",
    "https://en.wikipedia.org/wiki/Developing_country",
    "https://en.wikipedia.org/wiki/Small_Island_Developing_States",
    "https://en.wikipedia.org/wiki/Urban_sprawl",
    "https://en.wikipedia.org/wiki/Urbanization",
    "https://en.wikipedia.org/wiki/Rural_area",
    "https://en.wikipedia.org/wiki/Human_settlement",
    "https://en.wikipedia.org/wiki/Population_density",
    "https://en.wikipedia.org/wiki/Environmental_policy",
    "https://en.wikipedia.org/wiki/Environmental_governance",
    "https://en.wikipedia.org/wiki/Environmental_justice",
    "https://en.wikipedia.org/wiki/Climate_change_mitigation",
    "https://en.wikipedia.org/wiki/Climate_change_adaptation",
    "https://en.wikipedia.org/wiki/Sustainable_development",
    "https://en.wikipedia.org/wiki/Sustainable_cities",
    "https://en.wikipedia.org/wiki/Renewable_energy_transition",
    "https://en.wikipedia.org/wiki/Water_security",
    "https://en.wikipedia.org/wiki/Food_security",
    "https://en.wikipedia.org/wiki/Deforestation",
    "https://en.wikipedia.org/wiki/Land_degradation",
    "https://en.wikipedia.org/wiki/Environmental_degradation",
    "https://en.wikipedia.org/wiki/Biodiversity_loss",
    "https://en.wikipedia.org/wiki/Overpopulation",
    "https://en.wikipedia.org/wiki/Climate_migration",
    "https://en.wikipedia.org/wiki/Environmental_refugee",
    "https://en.wikipedia.org/wiki/Urban_heat_island",
    "https://en.wikipedia.org/wiki/Waste_management",
    "https://en.wikipedia.org/wiki/Air_pollution",
    "https://en.wikipedia.org/wiki/Regional_development",
    "https://en.wikipedia.org/wiki/Economic_geography",
    "https://en.wikipedia.org/wiki/Political_geography",
    "https://en.wikipedia.org/wiki/Geopolitics",
    "https://en.wikipedia.org/wiki/International_development",
    "https://en.wikipedia.org/wiki/Globalization",
    "https://en.wikipedia.org/wiki/Environmental_planning",
    "https://en.wikipedia.org/wiki/Urban_planning",
    "https://en.wikipedia.org/wiki/Land_use",
    "https://en.wikipedia.org/wiki/Smart_city"
  ],

  "medicine_and_biology": [
    "https://en.wikipedia.org/wiki/Biology",
    "https://en.wikipedia.org/wiki/Cell_biology",
    "https://en.wikipedia.org/wiki/Genetics",
    "https://en.wikipedia.org/wiki/Evolutionary_biology",
    "https://en.wikipedia.org/wiki/Molecular_biology",
    "https://en.wikipedia.org/wiki/Microbiology",
    "https://en.wikipedia.org/wiki/Neuroscience",
    "https://en.wikipedia.org/wiki/Ecology",
    "https://en.wikipedia.org/wiki/Botany",
    "https://en.wikipedia.org/wiki/Zoology",
    "https://en.wikipedia.org/wiki/Biochemistry",
    "https://en.wikipedia.org/wiki/DNA",
    "https://en.wikipedia.org/wiki/RNA",
    "https://en.wikipedia.org/wiki/Protein",
    "https://en.wikipedia.org/wiki/Metabolism",
    "https://en.wikipedia.org/wiki/Photosynthesis",
    "https://en.wikipedia.org/wiki/Mitosis",
    "https://en.wikipedia.org/wiki/Meiosis",
    "https://en.wikipedia.org/wiki/Crispr",
    "https://en.wikipedia.org/wiki/Symbiosis",
    "https://en.wikipedia.org/wiki/Medicine",
    "https://en.wikipedia.org/wiki/Anatomy",
    "https://en.wikipedia.org/wiki/Physiology",
    "https://en.wikipedia.org/wiki/Pharmacology",
    "https://en.wikipedia.org/wiki/Pathology",
    "https://en.wikipedia.org/wiki/Epidemiology",
    "https://en.wikipedia.org/wiki/Immunology",
    "https://en.wikipedia.org/wiki/Cardiology",
    "https://en.wikipedia.org/wiki/Oncology",
    "https://en.wikipedia.org/wiki/Neurology",
    "https://en.wikipedia.org/wiki/Endocrinology",
    "https://en.wikipedia.org/wiki/Psychiatry",
    "https://en.wikipedia.org/wiki/Surgery",
    "https://en.wikipedia.org/wiki/Vaccine",
    "https://en.wikipedia.org/wiki/Antibiotics",
    "https://en.wikipedia.org/wiki/Stem_cell",
    "https://en.wikipedia.org/wiki/Genomics",
    "https://en.wikipedia.org/wiki/Pediatrics",
    "https://en.wikipedia.org/wiki/Radiology",
    "https://en.wikipedia.org/wiki/Virology"
  ],

  "sports": [
    "https://en.wikipedia.org/wiki/Sport",
    "https://en.wikipedia.org/wiki/Association_football",
    "https://en.wikipedia.org/wiki/Cricket",
    "https://en.wikipedia.org/wiki/Basketball",
    "https://en.wikipedia.org/wiki/Baseball",
    "https://en.wikipedia.org/wiki/Volleyball",
    "https://en.wikipedia.org/wiki/Rugby_union",
    "https://en.wikipedia.org/wiki/American_football",
    "https://en.wikipedia.org/wiki/Tennis",
    "https://en.wikipedia.org/wiki/Badminton",
    "https://en.wikipedia.org/wiki/Table_tennis",
    "https://en.wikipedia.org/wiki/Swimming_(sport)",
    "https://en.wikipedia.org/wiki/Athletics_(sport)",
    "https://en.wikipedia.org/wiki/Gymnastics",
    "https://en.wikipedia.org/wiki/Boxing",
    "https://en.wikipedia.org/wiki/Wrestling",
    "https://en.wikipedia.org/wiki/Cycling",
    "https://en.wikipedia.org/wiki/Motorsport",
    "https://en.wikipedia.org/wiki/Formula_One",
    "https://en.wikipedia.org/wiki/Olympic_Games",
    "https://en.wikipedia.org/wiki/Paralympic_Games",
    "https://en.wikipedia.org/wiki/Commonwealth_Games",
    "https://en.wikipedia.org/wiki/FIFA_World_Cup",
    "https://en.wikipedia.org/wiki/ICC_Cricket_World_Cup",
    "https://en.wikipedia.org/wiki/Sports_science",
    "https://en.wikipedia.org/wiki/Sports_psychology",
    "https://en.wikipedia.org/wiki/Sports_nutrition",
    "https://en.wikipedia.org/wiki/Doping_in_sport",
    "https://en.wikipedia.org/wiki/Anti-doping",
    "https://en.wikipedia.org/wiki/Referee",
    "https://en.wikipedia.org/wiki/Coaching",
    "https://en.wikipedia.org/wiki/Sportsmanship",
    "https://en.wikipedia.org/wiki/Professional_sports",
    "https://en.wikipedia.org/wiki/Sports_league",
    "https://en.wikipedia.org/wiki/Sports_management",
    "https://en.wikipedia.org/wiki/Stadium",
    "https://en.wikipedia.org/wiki/Esports",
    "https://en.wikipedia.org/wiki/Women%27s_sports",
    "https://en.wikipedia.org/wiki/Youth_sports",
    "https://en.wikipedia.org/wiki/Sports_in_India"
  ]
}
