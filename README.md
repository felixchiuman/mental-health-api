# Mental Health API

A **FastAPI** project that leverages AI to summarize PTSD-related text, analyze emotions, and generate coping suggestions.

## Gradio Demo Code
You can access the gradio demo project here: [Mental Health Assistant](https://github.com/felixchiuman/mental-health-assistant)

## Features

- **Text Summarization**: AI-powered summarization of PTSD-related text.
- **Emotion Analysis**: Analyze emotions within a given text.
- **Coping Suggestions**: Generate personalized coping strategies based on the analysis.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/felixchiuman/mental-health-api.git
   cd mental-health-api

2. Set up a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Usage
1. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload

2. Access the API documentation:
   Open your browser and navigate to http://127.0.0.1:8000/docs for the Swagger UI.
