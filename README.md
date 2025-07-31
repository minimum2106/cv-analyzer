# CV Analyzer

CV Analyzer is a Streamlit-based web application that allows users to upload their CVs and job descriptions for AI-powered analysis. The app uses machine learning models to extract key information, generate job requirements, and compute similarity matrices to match CV lines with job descriptions.

---

## Features

- **Upload CVs**: Upload your CV in PDF format for analysis.
- **Job Description Input**: Add job descriptions or specific requirements for comparison.
- **AI-Powered Analysis**:
  - Extracts key lines and requirements from the CV and job description.
  - Computes similarity matrices to match CV lines with job requirements.
  - Provides explanations for matches using LLMs (e.g., OpenAI, Claude, Groq).
- **Interactive Visualization**: Highlights matched lines and provides detailed explanations.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- [Poetry](https://python-poetry.org/) (optional, for dependency management)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cv-analyzer.git
   cd cv-analyzer
   ```

2. Run application

```bash
    python -m backend.server # initiate backend
    streamlit run frontend/app.py # run frontend 
```

## Usage
- Upload Your CV
- Add Job Description
- Analyze: Click the "Analyze" button to generate job lines, compute similarity matrices, and view matches.
