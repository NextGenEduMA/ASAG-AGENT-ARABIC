from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import re
import google.generativeai as genai
import numpy as np
from sentence_transformers import SentenceTransformer, util
import uuid
from dotenv import load_dotenv
from pymongo import MongoClient
import datetime
import chromadb
from chromadb.utils import embedding_functions
import asyncio
import threading
from functools import lru_cache
import time
from concurrent.futures import ThreadPoolExecutor
from flask_caching import Cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", str(uuid.uuid4()))

# ========== CONFIGURATION ==========
app.config['CACHE_TYPE'] = 'simple'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300  # 5 minutes
cache = Cache(app)

# Thread pool for parallel operations
executor = ThreadPoolExecutor(max_workers=4)

# ========== MODEL INITIALIZATION ==========
@lru_cache(maxsize=1)
def get_sentence_transformer():
    """Cache the Sentence Transformer model"""
    try:
        return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    except Exception as e:
        logger.error(f"Error loading sentence transformer: {e}")
        return None

# Configure Google Gemini API
API_KEY = os.getenv("GOOGLE_API_KEY", "")
model = None

if not API_KEY:
    logger.error("No API key provided. Set GOOGLE_API_KEY in environment variables.")
else:
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
        logger.info("Gemini model initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Gemini model: {e}")

# ========== DATABASE INITIALIZATION ==========
MONGO_URI = os.getenv("MONGO_URI", "")
mongo_client = None
db = None
data_collection = None

if not MONGO_URI:
    logger.error("No MongoDB URI provided. Set MONGO_URI in environment variables.")
else:
    try:
        mongo_client = MongoClient(
            MONGO_URI,
            maxPoolSize=50,
            minPoolSize=5,
            maxIdleTimeMS=30000,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=20000
        )
        mongo_client.admin.command('ping')
        logger.info("MongoDB connection established successfully.")
        
        db = mongo_client['asag_arabic']
        data_collection = db['questions_collection']
        
        # Create indexes for performance
        data_collection.create_index("session_id")
        data_collection.create_index("created_at")
        logger.info("Database indexes created successfully.")
        
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")

# ========== CHROMADB INITIALIZATION ==========
chroma_client = None
collection = None

try:
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(
        name="arabic_educational_content",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-mpnet-base-v2"
        )
    )
    logger.info("ChromaDB initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing ChromaDB: {e}")

class ArabicEducationalAssistant:
    def __init__(self, genai_model, chroma_collection=None):
        self.model = genai_model
        self._sentence_model = None
        self.chroma_collection = chroma_collection
        logger.info("Educational Assistant initialized successfully!")

    @property
    def sentence_model(self):
        """Lazy loading of sentence transformer"""
        if self._sentence_model is None:
            self._sentence_model = get_sentence_transformer()
        return self._sentence_model

    @cache.memoize(timeout=600)
    def retrieve_context(self, query, k=3):
        """Retrieve relevant context from ChromaDB with caching"""
        if not self.chroma_collection:
            return ""
        try:
            results = self.chroma_collection.query(
                query_texts=[query],
                n_results=k
            )
            if results and 'documents' in results and results['documents'] and len(results['documents'][0]) > 0:
                return "\n\n".join(results['documents'][0])
            return ""
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return ""

    def add_to_knowledge_base(self, content, content_id):
        """Add content to ChromaDB knowledge base"""
        if not self.chroma_collection:
            return False
        
        def _add_chunks():
            try:
                chunks = []
                chunk_size = 500
                for i in range(0, len(content), chunk_size):
                    chunk = content[i:i+chunk_size]
                    if len(chunk.strip()) > 50:
                        chunks.append(chunk)
                
                documents = []
                metadatas = []
                ids = []
                
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({"content_id": content_id, "chunk_id": i})
                    ids.append(f"{content_id}_{i}")
                
                if documents:
                    self.chroma_collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                    logger.info(f"Added {len(chunks)} chunks to knowledge base for content_id: {content_id}")
                return True
            except Exception as e:
                logger.error(f"Error adding to knowledge base: {e}")
                return False
        
        executor.submit(_add_chunks)
        return True

    def generate_questions(self, content, num_questions=5, level=1):
        """Generate questions from content"""
        if not content or len(content.strip()) < 50:
            return [{"question": "يرجى تقديم نص أطول لتوليد أسئلة ذات معنى.", "answer": "النص المقدم قصير جدًا."}]
        
        content_id = str(uuid.uuid4())
        
        def get_context():
            if self.chroma_collection:
                self.add_to_knowledge_base(content, content_id)
                return self.retrieve_context(content)
            return ""
        
        context_future = executor.submit(get_context)
        
        class_mapping = {
            1: "الصف الأول (سهلة)",
            2: "الصف الثاني (سهلة)",
            3: "الصف الثالث (متوسطة)",
            4: "الصف الرابع (متوسطة)",
            5: "الصف الخامس (متقدمة)",
            6: "الصف السادس (متقدمة)"
        }
        
        difficulty_description = class_mapping.get(level, "الصف الأول (سهلة)")
        
        base_prompt = f"""
        أنشئ {num_questions} أسئلة مفتوحة مع إجابات لـ {difficulty_description}.

        النص:
        ---
        {content[:2000]}
        ---

        تنسيق الإجابة:
        1. السؤال: [السؤال هنا]
        الإجابة: [الإجابة هنا]

        2. السؤال: [السؤال هنا]
        الإجابة: [الإجابة هنا]
        """
        
        try:
            # Get context if available
            try:
                context = context_future.result(timeout=5)
                if context:
                    base_prompt = f"{base_prompt}\n\nسياق إضافي:\n{context[:500]}"
            except:
                pass
            
            response = self.model.generate_content(base_prompt)
            if not response or not hasattr(response, 'text') or not response.text:
                logger.error("Failed to generate questions. Empty response received.")
                return []
            
            qa_pairs = self.extract_qa_pairs(response.text)
            for qa in qa_pairs:
                qa["content_id"] = content_id
            return qa_pairs
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return []

    def extract_qa_pairs(self, text):
        """Extract question-answer pairs from generated text"""
        qa_pairs = []
        
        pattern = r'(\d+)\.\s*السؤال:\s*(.*?)\s*الإجابة:\s*(.*?)(?=\d+\.\s*السؤال:|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            question = match[1].strip()
            answer = match[2].strip()
            if question and answer:
                qa_pairs.append({"question": question, "answer": answer})
        
        # Fallback extraction method
        if not qa_pairs:
            lines = text.split('\n')
            current_q = ""
            current_a = ""
            in_answer = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if 'السؤال:' in line:
                    if current_q and current_a:
                        qa_pairs.append({"question": current_q, "answer": current_a})
                    current_q = line.replace('السؤال:', '').strip()
                    current_a = ""
                    in_answer = False
                elif 'الإجابة:' in line:
                    current_a = line.replace('الإجابة:', '').strip()
                    in_answer = True
                elif in_answer:
                    current_a += " " + line
                elif current_q and not in_answer:
                    current_q += " " + line
            
            if current_q and current_a:
                qa_pairs.append({"question": current_q, "answer": current_a})
        
        return qa_pairs

    @cache.memoize(timeout=300)
    def calculate_similarity(self, text1, text2):
        """Calculate similarity with caching"""
        if not self.sentence_model:
            return 0.0
        try:
            embedding1 = self.sentence_model.encode(text1, convert_to_tensor=True)
            embedding2 = self.sentence_model.encode(text2, convert_to_tensor=True)
            return util.pytorch_cos_sim(embedding1, embedding2).item()
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def evaluate_answer(self, reference_answer, student_answer):
        """Evaluate student answer against reference answer"""
        if not student_answer or len(student_answer.strip()) < 5:
            return {
                "score": 0,
                "feedback": "الإجابة قصيرة جدًا أو غير موجودة.",
                "detailed_feedback": "يرجى تقديم إجابة مفصلة تحتوي على معلومات واضحة."
            }

        def normalize_text(text):
            return re.sub(r'[،؛؟.!]', '', re.sub(r'\s+', ' ', text.strip().lower()))

        normalized_ref = normalize_text(reference_answer)
        normalized_stu = normalize_text(student_answer)
        
        if normalized_ref == normalized_stu:
            return {
                "score": 10,
                "feedback": "ممتاز! إجابتك دقيقة ومتكاملة!",
                "detailed_feedback": "لقد قدمت إجابة مطابقة تمامًا للإجابة النموذجية.",
                "similarity": 1.0
            }

        similarity = self.calculate_similarity(reference_answer, student_answer)

        if similarity > 0.95:
            return {
                "score": 10,
                "feedback": "ممتاز! إجابتك دقيقة ومتكاملة!",
                "detailed_feedback": "لقد قدمت إجابة مطابقة تقريبًا للإجابة النموذجية.",
                "similarity": similarity
            }

        return self.generate_evaluation_feedback(similarity)

    def generate_evaluation_feedback(self, similarity):
        """Generate evaluation feedback based on similarity score"""
        score = max(0, min(10, round(similarity * 10)))
        
        feedback_map = {
            (0.8, 1.0): ("ممتاز! إجابتك دقيقة ومتكاملة!", "إجابة ممتازة تغطي النقاط الأساسية بشكل شامل."),
            (0.6, 0.8): ("جيد جدًا! إجابتك تحتوي على معلومات مهمة.", "إجابة جيدة، يمكن إضافة المزيد من التفاصيل."),
            (0.4, 0.6): ("جيد! إجابتك تحتوي على بعض النقاط الصحيحة.", "إجابة في الطريق الصحيح، حاول إضافة المزيد من المعلومات."),
            (0.2, 0.4): ("حاول مرة أخرى! إجابتك تحتاج إلى مزيد من التفاصيل.", "راجع النص وأضف معلومات أكثر تفصيلاً."),
            (0.0, 0.2): ("الإجابة غير متعلقة بالسؤال.", "راجع السؤال والنص بعناية وحاول الإجابة بدقة أكبر.")
        }
        
        for (min_sim, max_sim), (feedback, detailed) in feedback_map.items():
            if min_sim <= similarity < max_sim:
                return {
                    "score": score,
                    "feedback": feedback,
                    "detailed_feedback": detailed,
                    "similarity": similarity
                }
        
        return {
            "score": 0,
            "feedback": "يرجى المحاولة مرة أخرى",
            "detailed_feedback": "راجع السؤال والنص وحاول تقديم إجابة أكثر دقة.",
            "similarity": similarity
        }

# Initialize assistant
assistant = None
if model:
    assistant = ArabicEducationalAssistant(model, collection)
else:
    logger.error("Assistant not initialized due to missing Gemini model.")

# ========== ROUTES ==========
@app.route('/')
@cache.cached(timeout=300)
def index():
    """Main page route"""
    if not assistant:
        flash("ERROR: System not initialized. Please check API key configuration.", "error")
    if mongo_client is None:
        flash("ERROR: MongoDB not initialized. Please check MONGO_URI configuration.", "error")
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Generate questions route"""
    start_time = time.time()
    
    if not assistant:
        flash("System not initialized. Please check API key configuration.", "error")
        return redirect(url_for('index'))
    
    if mongo_client is None or data_collection is None:
        flash("ERROR: MongoDB connection failed.", "error")
        return redirect(url_for('index'))

    try:
        content = ""
        input_type = request.form.get('input_type')
        
        if input_type == 'text':
            content = request.form.get('content', '').strip()
        elif input_type == 'file':
            if 'file' not in request.files or request.files['file'].filename == '':
                flash("No file uploaded or selected.", "error")
                return redirect(url_for('index'))
            
            file = request.files['file']
            try:
                content = file.read().decode('utf-8').strip()
            except UnicodeDecodeError:
                try:
                    file.seek(0)
                    content = file.read().decode('cp1256').strip()
                except Exception as e:
                    flash(f"Error reading file: {e}", "error")
                    return redirect(url_for('index'))

        if len(content) < 50:
            flash("Insufficient content provided. Please provide more text.", "error")
            return redirect(url_for('index'))

        try:
            num_questions = max(1, min(10, int(request.form.get('num_questions', 5))))
            level = max(1, min(6, int(request.form.get('level', 1))))
        except ValueError:
            flash("Invalid parameters.", "error")
            return redirect(url_for('index'))

        try:
            qa_pairs = assistant.generate_questions(content, num_questions, level)
            if not qa_pairs:
                flash("Failed to generate questions. Please try again.", "error")
                return redirect(url_for('index'))

            session_id = str(uuid.uuid4())
            
            def save_to_db():
                try:
                    questions_doc = {
                        "session_id": session_id,
                        "content": content[:1000],
                        "questions": qa_pairs,
                        "level": level,
                        "created_at": datetime.datetime.utcnow()
                    }
                    data_collection.insert_one(questions_doc)
                    logger.info(f"Document saved for session: {session_id}")
                except Exception as e:
                    logger.error(f"MongoDB insertion error: {e}")
            
            executor.submit(save_to_db)
            
            logger.info(f"Generation completed in {time.time() - start_time:.2f} seconds")
            flash("Questions generated successfully!", "success")
            return render_template('questions.html', questions=qa_pairs, session_id=session_id, level=level)
            
        except Exception as e:
            flash(f"Error generating questions: {e}", "error")
            return redirect(url_for('index'))

    except Exception as e:
        flash(f"Unexpected error: {e}", "error")
        return redirect(url_for('index'))

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """Evaluate student answer route"""
    start_time = time.time()
    
    if not assistant or mongo_client is None or data_collection is None:
        return jsonify({"error": "System not initialized."}), 500

    session_id = request.form.get('session_id', '').strip()
    student_answer = request.form.get('student_answer', '').strip()
    
    try:
        question_index = int(request.form.get('question_index', -1))
    except ValueError:
        return jsonify({"error": "Invalid question index."}), 400

    if len(student_answer) < 5:
        return jsonify({"error": "Answer is too short."}), 400

    try:
        session_doc = data_collection.find_one(
            {"session_id": session_id},
            {"questions": 1}
        )
        
        if not session_doc or question_index >= len(session_doc['questions']):
            return jsonify({"error": "Invalid session or question."}), 400

        reference_answer = session_doc['questions'][question_index]['answer']
        evaluation = assistant.evaluate_answer(reference_answer, student_answer)
        
        def save_evaluation():
            try:
                response_doc = {
                    "session_id": session_id,
                    "question_index": question_index,
                    "student_answer": student_answer,
                    "evaluation": evaluation,
                    "created_at": datetime.datetime.utcnow()
                }
                data_collection.insert_one(response_doc)
            except Exception as e:
                logger.error(f"Error saving evaluation: {e}")
        
        executor.submit(save_evaluation)
        
        logger.info(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
        return jsonify(evaluation)
        
    except Exception as e:
        logger.error(f"Error evaluating answer: {e}")
        return jsonify({"error": "Evaluation failed."}), 500

# ========== CLEANUP ==========
import atexit

def cleanup():
    """Cleanup resources on application shutdown"""
    if mongo_client:
        mongo_client.close()
    executor.shutdown(wait=True)

atexit.register(cleanup)

if __name__ == '__main__':
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)