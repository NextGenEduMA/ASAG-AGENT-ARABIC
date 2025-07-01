# ... (all imports and setup code are correct) ...
from flask import Flask, jsonify, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv
import uuid
import time
from datetime import datetime
import json
import re
import os
from functools import lru_cache
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from pymongo import MongoClient
from bson import ObjectId
import chromadb
from chromadb.utils import embedding_functions
import logging
from flask_caching import Cache
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "a-default-secret-key-for-dev")
app.config['CACHE_TYPE'] = 'simple'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300
cache = Cache(app)

@app.template_filter('strftime')
def _jinja2_filter_datetime(unix_timestamp, fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d %H:%M'
    return datetime.fromtimestamp(unix_timestamp).strftime(fmt)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.warning("CRITICAL: GOOGLE_API_KEY not found in .env file.")

generative_model = None
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        generative_model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
        logger.info("Gemini API initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini API: {e}")

questions_collection = None
evaluation_collection = None
try:
    mongo_uri = os.getenv('MONGO_URI')
    if not mongo_uri:
        mongo_uri = 'mongodb://localhost:27017/question_generator'
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client.get_default_database()
    if not db.name:
         raise ValueError("MongoDB URI must include a database name.")
    questions_collection = db['questions']
    evaluation_collection = db['evaluations']
    mongo_client.server_info()
    logger.info(f"MongoDB connection successful to database: '{db.name}'.")
except Exception as e:
    logger.error(f"Failed to initialize MongoDB: {e}")

chroma_collection = None
if GOOGLE_API_KEY:
    try:
        gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GOOGLE_API_KEY)
        chroma_client = chromadb.Client()
        chroma_collection = chroma_client.get_or_create_collection(name='question_vectors_google', embedding_function=gemini_ef)
        logger.info("ChromaDB client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")

@lru_cache(maxsize=1)
def get_sentence_transformer():
    logger.info("Loading local Sentence Transformer model...")
    try:
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        logger.info("Sentence Transformer model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"FATAL: Error loading sentence transformer: {e}")
        return None
similarity_model = get_sentence_transformer()

class QuestionAssistant:
    def __init__(self, model):
        self.model = model
    @lru_cache(maxsize=100)
    def generate_questions(self, content, num_questions, level):
        if not self.model: 
            logger.error("Model not initialized")
            return []
        
        prompt = f"""
        Generate exactly {num_questions} question-answer pairs based on the following content, suitable for a level {level} student.
        Return ONLY a valid JSON array of objects. Each object must have a "question" key and an "answer" key.
        Do not include any introductory text, concluding remarks, or markdown formatting like ```json.

        Content:
        ---
        {content[:2000]}
        ---
        """
        
        try:
            response = self.model.generate_content(prompt)
            
            # Log the raw response for debugging
            logger.info(f"Raw response from Gemini: {response.text[:200]}...")
            
            if not response.text or not response.text.strip():
                logger.error("Empty response from Gemini API")
                return []
            
            cleaned_text = response.text.strip().replace('```json', '').replace('```', '').strip()
            
            # Log the cleaned text for debugging
            logger.info(f"Cleaned text: {cleaned_text[:200]}...")
            
            if not cleaned_text:
                logger.error("Empty cleaned text after processing")
                return []
            
            qa_pairs = json.loads(cleaned_text)
            
            # Validate the structure
            if not isinstance(qa_pairs, list):
                logger.error(f"Response is not a list: {type(qa_pairs)}")
                return []
            
            if not all(isinstance(item, dict) and 'question' in item and 'answer' in item for item in qa_pairs):
                logger.error(f"Invalid structure in response: {qa_pairs}")
                return []
            
            return qa_pairs
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}. Response text: '{response.text[:200]}...'")
            return []
        except Exception as e:
            logger.error(f"Error in generate_questions: {e}", exc_info=True)
            return []
assistant = QuestionAssistant(generative_model)


@app.route('/')
def index():
    return render_template('index.html')

# --- MODIFICATION 1: Pass the full 'content' to the preview template ---
@app.route('/generate', methods=['POST'])
def generate():
    if not assistant.model:
        flash("System not initialized.", "error")
        return redirect(url_for('index'))
    try:
        content = request.form.get('content', '').strip()
        num_questions = int(request.form.get('num_questions', 5))
        level = int(request.form.get('level', 1))
        if len(content) < 50:
            flash("Insufficient content provided.", "error")
            return redirect(url_for('index'))
        qa_pairs = assistant.generate_questions(content, num_questions, level)
        if not qa_pairs:
            flash("Failed to generate questions.", "error")
            return redirect(url_for('index'))
        preview_id = str(uuid.uuid4())
        flash("Questions generated successfully!", "success")
        return render_template('preview.html', 
                             questions=qa_pairs, 
                             preview_id=preview_id, 
                             level=level,
                             content=content) # Pass the full content here
    except Exception as e:
        logger.error(f"Unexpected error in /generate: {e}", exc_info=True)
        flash("An unexpected server error occurred.", "error")
        return redirect(url_for('index'))

# --- MODIFICATION 2: Save the 'original_content' to the database ---
@app.route('/save_questions', methods=['POST'])
def save_questions():
    try:
        preview_id = request.form.get('preview_id')
        questions_str = request.form.get('questions', '[]')
        questions = json.loads(questions_str)
        original_content = request.form.get('original_content', '') # Get content from form

        if not preview_id or not questions:
            flash("No questions provided to save.", "error")
            return redirect(url_for('index'))
        
        if questions_collection is not None:
            questions_collection.insert_one({
                'preview_id': preview_id,
                'questions': questions,
                'timestamp': time.time(),
                'content': original_content  # Save the content
            })
            logger.info(f"Saved {len(questions)} questions and content to MongoDB.")
        
        if chroma_collection is not None:
            try:
                ids = [str(uuid.uuid4()) for _ in questions]
                docs = [qa['question'] for qa in questions]
                metadatas = [{'answer': qa['answer']} for qa in questions]
                logger.info(f"Attempting to add {len(questions)} questions to ChromaDB...")
                chroma_collection.add(ids=ids, documents=docs, metadatas=metadatas)
                logger.info(f"Successfully added {len(questions)} questions to ChromaDB.")
            except Exception as e:
                logger.error(f"Failed to add questions to ChromaDB: {e}")
        else:
            logger.warning("ChromaDB collection is None - questions not saved to ChromaDB")
            
        flash("Questions saved successfully!", "success")
        return redirect(url_for('questions'))

    except Exception as e:
        logger.error(f"Error saving questions: {e}", exc_info=True)
        flash(f"Error saving questions: {e}", "error")
        return redirect(url_for('index'))

# ... (rest of your routes are correct) ...
@app.route('/questions')
def questions():
    if questions_collection is None:
        flash("Database not initialized.", "error")
        return render_template('questions.html', question_sets=[])
    question_sets = list(questions_collection.find().sort('timestamp', -1).limit(10))
    return render_template('questions.html', question_sets=question_sets)

@app.route('/evaluate_set/<set_id>')
def evaluate_set(set_id):
    if questions_collection is None:
        flash("Database not initialized.", "error")
        return redirect(url_for('questions'))
    try:
        question_set = questions_collection.find_one({"_id": ObjectId(set_id)})
        if not question_set:
            flash("Question set not found.", "error")
            return redirect(url_for('questions'))
        return render_template('evaluate.html', question_set=question_set)
    except Exception as e:
        logger.error(f"Error fetching question set for evaluation: {e}")
        flash("Could not load the question set.", "error")
        return redirect(url_for('questions'))

@app.route('/evaluate_answer', methods=['POST'])
def evaluate_answer():
    if not similarity_model:
        return jsonify({"error": "خدمة التقييم غير متوفرة حاليًا."}), 503
    
    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data received in evaluate_answer")
            return jsonify({"error": "لم يتم استلام بيانات."}), 400
            
        student_answer = data.get('student_answer', '').strip()
        model_answer = data.get('model_answer', '').strip()
        question = data.get('question', '').strip()
        
        logger.info(f"Evaluation request - Question: {question[:50]}..., Student: {student_answer[:50]}..., Model: {model_answer[:50]}...")
        
        if not student_answer or not model_answer or not question:
            logger.error(f"Missing data - Student: {bool(student_answer)}, Model: {bool(model_answer)}, Question: {bool(question)}")
            return jsonify({"error": "بيانات التقييم غير كاملة."}), 400
        
        # Use Gemini for both scoring and feedback
        final_score = None
        feedback = None
        
        if assistant.model:
            prompt = f"""
            You are an educational assessment expert. Evaluate the student's answer compared to the model answer for the given question.
            
            Provide your response in the following JSON format:
            {{
                "score": [number from 0 to 10],
                "feedback": "[detailed feedback in Arabic]"
            }}
            
            Scoring criteria:
            - 9-10: Excellent, complete and accurate answer
            - 7-8: Good answer with minor gaps
            - 5-6: Adequate answer but missing key points
            - 3-4: Partial answer with significant gaps
            - 0-2: Incorrect or irrelevant answer
            
            Question: {question}
            Student Answer: {student_answer}
            Model Answer: {model_answer}
            
            Provide encouraging and constructive feedback in Arabic. Focus on what the student did well and suggest specific improvements.
            """
            try:
                response = assistant.model.generate_content(prompt)
                response_text = response.text.strip()
                
                # Try to parse JSON response
                import re
                json_match = re.search(r'\{[^{}]*"score"[^{}]*"feedback"[^{}]*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    evaluation_data = json.loads(json_str)
                    final_score = float(evaluation_data.get('score', 0))
                    feedback = evaluation_data.get('feedback', '')
                else:
                    # If JSON parsing fails, try to extract score and feedback manually
                    score_match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', response_text)
                    feedback_match = re.search(r'"feedback"\s*:\s*"([^"]+)"', response_text)
                    
                    if score_match:
                        final_score = float(score_match.group(1))
                    if feedback_match:
                        feedback = feedback_match.group(1)
                        
                logger.info(f"Gemini evaluation - Score: {final_score}, Feedback: {feedback[:100] if feedback else 'None'}...")
                
            except Exception as e:
                logger.error(f"Error getting evaluation from Gemini: {e}")
                final_score = None
                feedback = None
        
        # Fallback to local similarity model if Gemini fails
        if final_score is None or feedback is None:
            logger.warning("Falling back to local similarity model")
            embeddings = similarity_model.encode([student_answer, model_answer])
            similarity_score = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0]
            final_score = round(float(similarity_score) * 10, 1)
            
            if final_score >= 8.5:
                feedback = "إجابة ممتازة ومطابقة للمعنى المطلوب."
            elif final_score >= 6.5:
                feedback = "إجابة جيدة، تحتوي على النقاط الأساسية."
            elif final_score >= 4.0:
                feedback = "إجابتك صحيحة جزئياً، لكنها تفتقد بعض التفاصيل المهمة."
            else:
                feedback = "الإجابة لا تتطابق بشكل كبير مع الجواب النموذجي. حاول مرة أخرى."
        
        # Ensure score is within valid range
        final_score = max(0, min(10, final_score))
        
        # Log the evaluation
        logger.info(f"Final evaluation - Score: {final_score}, Feedback: {feedback[:100]}...")
        
        return jsonify({"score": final_score, "feedback": feedback})
    
    except Exception as e:
        logger.error(f"Error in /evaluate_answer: {e}", exc_info=True)
        return jsonify({"error": "حدث خطأ داخلي أثناء التقييم."}), 500

@app.route('/save_evaluation', methods=['POST'])
def save_evaluation():
    if evaluation_collection is None:
        flash("Database not configured to save evaluations.", "error")
        return redirect(url_for('questions'))
    try:
        question_set_id = request.form.get('question_set_id')
        evaluation_data_str = request.form.get('evaluation_data')
        if not question_set_id or not evaluation_data_str:
            flash("Incomplete evaluation data received.", "error")
            return redirect(url_for('questions'))
        evaluation_data = json.loads(evaluation_data_str)
        total_score = sum(item['score'] for item in evaluation_data)
        average_score = total_score / len(evaluation_data) if evaluation_data else 0
        evaluation_collection.insert_one({
            "original_question_set_id": ObjectId(question_set_id),
            "evaluation_results": evaluation_data,
            "average_score": round(average_score, 2),
            "timestamp": time.time()
        })
        flash(f"Evaluation saved successfully! Average score: {average_score:.2f}/10", "success")
        return redirect(url_for('questions'))
    except Exception as e:
        logger.error(f"Error saving evaluation: {e}", exc_info=True)
        flash("An error occurred while saving the evaluation.", "error")
        return redirect(url_for('questions'))

@app.route('/test')
def test():
    return jsonify({"status": "working"})

@app.route('/chromadb')
def view_chromadb():
    if chroma_collection is None:
        return jsonify({"error": "ChromaDB not initialized"}), 503
    
    try:
        # Get all data from ChromaDB
        result = chroma_collection.get()
        
        data = {
            "collection_name": chroma_collection.name,
            "total_items": len(result['ids']),
            "items": []
        }
        
        for i in range(len(result['ids'])):
            item = {
                "id": result['ids'][i],
                "document": result['documents'][i],
                "metadata": result['metadatas'][i] if result['metadatas'] else None
            }
            data["items"].append(item)
        
        return jsonify(data)
    
    except Exception as e:
        logger.error(f"Error retrieving ChromaDB data: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)