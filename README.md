# Arabic Educational Assistant with RAG

This application is an intelligent educational tool designed to generate and evaluate educational questions in Arabic. It uses Google's Gemini AI model to create customized educational questions based on provided text content and evaluate student answers.

## License

Developed by Siham EL kouaze

## Features

- Generate open-ended questions with model answers from educational texts
- Support for different difficulty levels (grades 3-6)
- Evaluate student answers with detailed feedback
- Web interface with Arabic language support
- RAG (Retrieval-Augmented Generation) capabilities for enhanced question generation

## How to Use RAG

RAG (Retrieval-Augmented Generation) enhances question generation by retrieving relevant information from your knowledge base. To use RAG:

1. Place your educational materials (PDF or TXT files) in the `Texts` folder
2. The system will automatically create a vector database from these materials
3. When generating questions, the system will retrieve relevant passages to enhance the questions and answers

## Setup Instructions

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up your environment variables in `.env`:
   ```
   GOOGLE_API_KEY=your_google_api_key
   MONGO_URI=your_mongodb_connection_string
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Access the web interface at `http://localhost:5000`

## Using RAG Helper

To use the RAG functionality in your application:

```python
from rag_helper import get_rag_assistant

# Get the RAG assistant
rag = get_rag_assistant()

# Generate questions with RAG
questions = rag.generate_rag_questions(content, num_questions=5, level=2)
```

## Requirements

- Python 3.8+
- Google Gemini API key
- MongoDB (local or Atlas)
- Educational materials in PDF or TXT format (for RAG functionality)