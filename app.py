from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import re
import google.generativeai as genai
import numpy as np
from sentence_transformers import SentenceTransformer, util
import uuid
from dotenv import load_dotenv
from pymongo import MongoClient
from bson.objectid import ObjectId
import datetime

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = str(uuid.uuid4())

# Configure Google Gemini API
API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not API_KEY:
    print("ERROR: No API key provided. Set GOOGLE_API_KEY in environment variables.")
else:
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
        print("Gemini model initialized successfully.")
    except Exception as e:
        print(f"ERROR initializing Gemini model: {e}")
        model = None

# Configure MongoDB Atlas connection
MONGO_URI = os.getenv("MONGO_URI", "")
if not MONGO_URI:
    print("ERROR: No MongoDB URI provided. Set MONGO_URI in environment variables.")
    mongo_client = None
else:
    try:
        mongo_client = MongoClient(MONGO_URI)
        # Ping the database to verify connection
        mongo_client.admin.command('ping')
        print("MongoDB connection established successfully.")
        db = mongo_client['asag_arabic']
        data_collection = db['questions_collection']
        print(f"data_collection initialized: {data_collection is not None}")
    except Exception as e:
        print(f"ERROR connecting to MongoDB: {e}")
        mongo_client = None
        db = None
        data_collection = None

class ArabicEducationalAssistant:
    def __init__(self, genai_model):
        self.model = genai_model
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        print("Educational Assistant initialized successfully!")

    def generate_questions(self, content, num_questions=5, level=1):
        if not content or len(content.strip()) < 50:
            return [{"question": "يرجى تقديم نص أطول لتوليد أسئلة ذات معنى.", "answer": "النص المقدم قصير جدًا."}]

        class_mapping = {
            1: "الصف الثالث (سهلة وبسيطة)",
            2: "الصف الرابع (بسيطة مع تفكير بسيط)",
            3: "الصف الخامس (متوسطة مع تحليل)",
            4: "الصف السادس (معقدة مع تفكير نقدي)"
        }
        difficulty_description = class_mapping.get(level, "الصف الثالث (سهلة وبسيطة)")

        if level == 1:
            complexity_instruction = "الأسئلة يجب أن تكون قصيرة وسهلة جدًا، مناسبة لطلاب الصف الثالث. استخدم كلمات بسيطة جدًا وتأكد أن الإجابات تتطلب استرجاع معلومات مباشرة من النص."
        elif level == 2:
            complexity_instruction = "الأسئلة يجب أن تكون بسيطة ولكن تتطلب تفكيرًا بسيطًا، مناسبة لطلاب الصف الرابع. استخدم كلمات بسيطة مع التركيز على أسئلة تشرح 'لماذا' أو 'كيف' بشكل مباشر."
        elif level == 3:
            complexity_instruction = "الأسئلة يجب أن تكون متوسطة التعقيد، مناسبة لطلاب الصف الخامس. ركز على أسئلة تتطلب التحليل أو المقارنة، مع استخدام كلمات مناسبة لمستوى الفهم هذا."
        else:
            complexity_instruction = "الأسئلة يجب أن تكون معقدة وتتطلب تفكيرًا نقديًا، مناسبة لطلاب الصف السادس. ركز على أسئلة تتطلب استنتاج الأسباب والنتائج، التطبيقات العملية، أو المقارنات العميقة."

        prompt = f"""
        أنت مساعد تعليمي متخصص في اللغة العربية. مهمتك هي قراءة النص التالي وإنشاء {num_questions} أسئلة **مفتوحة** مع إجاباتها النموذجية لـ {difficulty_description}.

        يجب أن تتطلب هذه الأسئلة تفكيرًا وفهمًا للنص، وليس مجرد استرجاع مباشر للمعلومات.
        {complexity_instruction}
        تجنب الأسئلة التي يمكن الإجابة عليها بـ 'نعم' أو 'لا' أو بكلمة واحدة.

        لكل سؤال، قدم:
        1. السؤال
        2. الإجابة النموذجية المفصلة

        النص:
        ---
        {content}
        ---

        أسِئلة وإجابات:
        """

        try:
            response = self.model.generate_content(prompt)
            if not response or not hasattr(response, 'text') or not response.text:
                print("Failed to generate questions. Empty response received.")
                return []

            qa_pairs = self.extract_qa_pairs(response.text)
            return qa_pairs
        except Exception as e:
            print(f"Error generating questions: {e}")
            return []

    def extract_qa_pairs(self, text):
        qa_pairs = []
        lines = text.split('\n')
        current_q = ""
        current_a = ""
        qa_state = None

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if re.match(r'^\d+[\.\)]', line) or "سؤال" in line.lower() or ":" in line and not current_q:
                if current_q and current_a:
                    qa_pairs.append({"question": current_q, "answer": current_a})
                current_q = line
                current_a = ""
                qa_state = "question"
            elif re.match(r'^الإجابة|^إجابة|^الجواب', line) or qa_state == "question" and ":" in line:
                if ":" in line and qa_state == "question":
                    current_a = line
                    qa_state = "answer"
                else:
                    current_a += " " + line
            else:
                if qa_state == "question":
                    current_q += " " + line
                elif qa_state == "answer":
                    current_a += " " + line

        if current_q and current_a:
            qa_pairs.append({"question": current_q, "answer": current_a})

        if not qa_pairs:
            q_blocks = re.split(r'\d+[\.\)]', text)
            if len(q_blocks) > 1:
                q_blocks = q_blocks[1:]
                for block in q_blocks:
                    parts = re.split(r'الإجابة|إجابة|الجواب|:', block, 1)
                    if len(parts) > 1:
                        qa_pairs.append({"question": parts[0].strip(), "answer": parts[1].strip()})
                    else:
                        qa_pairs.append({"question": block.strip(), "answer": "يرجى الرجوع إلى النص لتحديد الإجابة المناسبة."})

        return qa_pairs

    def evaluate_answer(self, reference_answer, student_answer):
        if not student_answer or len(student_answer.strip()) < 5:
            return {
                "score": 1,
                "feedback": "الإجابة قصيرة جدًا أو غير مكتملة.",
                "detailed_feedback": "يرجى تقديم إجابة أكثر تفصيلاً."
            }

        reference_embedding = self.sentence_model.encode(reference_answer, convert_to_tensor=True)
        student_embedding = self.sentence_model.encode(student_answer, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(reference_embedding, student_embedding).item()

        prompt = f"""
        أنت مقيّم تعليمي متخصص في تقييم إجابات الطلاب.

        الإجابة النموذجية:
        ---
        {reference_answer}
        ---

        إجابة الطالب:
        ---
        {student_answer}
        ---

        قم بتقييم إجابة الطالب على مقياس من 1 إلى 10 نقاط، حيث:
        1-2 = غير مقبول
        3-4 = ضعيف
        5-6 = متوسط
        7-8 = جيد
        9-10 = ممتاز

        قدم النتيجة في السطر الأول بالصيغة التالية: "النتيجة: X/10"

        ثم قدم تقييمًا مفصلاً موجزًا (3-5 أسطر) يشرح نقاط القوة والضعف في إجابة الطالب.
        كن إيجابي ومشجعًا وقدم اقتراحات محددة للتحسين.
        """

        try:
            response = self.model.generate_content(prompt)
            if not response or not hasattr(response, 'text') or not response.text:
                return self.generate_fallback_evaluation(similarity)

            evaluation_text = response.text.strip()
            score_match = re.search(r'النتيجة:\s*(\d+)[/\.]10', evaluation_text)
            score = int(score_match.group(1)) if score_match else round(similarity * 10)
            feedback_lines = evaluation_text.split('\n')
            feedback = feedback_lines[0]
            detailed_feedback = '\n'.join(feedback_lines[1:]) if len(feedback_lines) > 1 else "تحتاج إلى مزيد من التفاصيل."

            return {
                "score": score,
                "feedback": feedback,
                "detailed_feedback": detailed_feedback,
                "similarity": similarity
            }
        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return self.generate_fallback_evaluation(similarity)

    def generate_fallback_evaluation(self, similarity):
        score = max(1, min(5, round(similarity * 5)))
        if similarity > 0.8:
            feedback = "ممتاز! إجابتك صحيحة ومكتملة."
        elif similarity > 0.6:
            feedback = "جيد جدا! إجابتك صحيحة لكنها تحتاج إلى بعض التفاصيل الإضافية."
        elif similarity > 0.4:
            feedback = "متوسط. إجابتك تحتوي على بعض المعلومات الصحيحة لكنها غير مكتملة."
        elif similarity > 0.2:
            feedback = "ضعيف. إجابتك بحاجة إلى مزيد من العمل والتفاصيل."
        else:
            feedback = "غير مقبول. حاول مرة أخرى بعد مراجعة المادة."

        return {
            "score": score,
            "feedback": feedback,
            "detailed_feedback": f"تم تقييم الإجابة باستخدام مقارنة النصوص. مستوى التشابه: {similarity:.2f}",
            "similarity": similarity
        }

# Initialize assistant
if 'model' in globals() and model:
    assistant = ArabicEducationalAssistant(model)
else:
    assistant = None
    print("ERROR: Assistant not initialized due to missing Gemini model.")

@app.route('/')
def index():
    if not assistant:
        flash("ERROR: System not initialized. Please check API key configuration.", "error")
    if mongo_client is None:
        flash("ERROR: MongoDB not initialized. Please check MONGO_URI configuration.", "error")
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if not assistant:
        flash("System not initialized. Please check API key configuration.", "error")
        return redirect(url_for('index'))
    
    # Check MongoDB connection
    print(f"mongo_client: {mongo_client}, data_collection: {data_collection}")
    if mongo_client is None or data_collection is None:
        flash("ERROR: MongoDB connection failed. Please check MONGO_URI configuration.", "error")
        return redirect(url_for('index'))

    try:
        mongo_client.admin.command('ping')
    except Exception as e:
        flash(f"MongoDB connection error: {e}", "error")
        return redirect(url_for('index'))

    content = ""
    input_type = request.form.get('input_type')

    if input_type == 'text':
        content = request.form.get('content', '')
    elif input_type == 'file':
        if 'file' not in request.files:
            flash("No file uploaded.", "error")
            return redirect(url_for('index'))
        file = request.files['file']
        if file.filename == '':
            flash("No file selected.", "error")
            return redirect(url_for('index'))
        try:
            content = file.read().decode('utf-8')
        except UnicodeDecodeError:
            try:
                content = file.read().decode('cp1256')
            except Exception as e:
                flash(f"Error reading file: {e}", "error")
                return redirect(url_for('index'))

    if not content or len(content.strip()) < 50:
        flash("Insufficient content provided. Please provide more text.", "error")
        return redirect(url_for('index'))

    try:
        num_questions = int(request.form.get('num_questions', 5))
        level = int(request.form.get('level', 1))
        if level < 1 or level > 4:
            flash("Invalid level selected. Please choose a class between 3rd and 6th.", "error")
            return redirect(url_for('index'))

        qa_pairs = assistant.generate_questions(content, num_questions, level)
        if not qa_pairs:
            flash("Failed to generate questions. Please try again.", "error")
            return redirect(url_for('index'))

        # Store questions in MongoDB
        session_id = str(uuid.uuid4())
        try:
            questions_doc = {
                "session_id": session_id,
                "content": content,
                "questions": qa_pairs,
                "level": level,
                "created_at": datetime.datetime.utcnow()
            }
            result = data_collection.insert_one(questions_doc)
            print(f"Document inserted with ID: {result.inserted_id}")
        except Exception as e:
            print(f"MongoDB insertion error: {e}")
            flash(f"Database error: {e}", "error")
            return redirect(url_for('index'))

        flash("Questions generated successfully!", "success")
        return render_template('questions.html', questions=qa_pairs, session_id=session_id, level=level)
    except Exception as e:
        flash(f"Error generating questions: {e}", "error")
        return redirect(url_for('index'))

@app.route('/evaluate', methods=['POST'])
def evaluate():
    if not assistant:
        return jsonify({"error": "System not initialized."}), 500
    
    if mongo_client is None or data_collection is None:
        return jsonify({"error": "MongoDB connection failed."}), 500

    try:
        mongo_client.admin.command('ping')
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        return jsonify({"error": f"MongoDB connection error: {e}"}), 500

    session_id = request.form.get('session_id', '')
    question_index = request.form.get('question_index', -1)
    student_answer = request.form.get('student_answer', '')

    print(f"Received: session_id={session_id}, question_index={question_index}, student_answer={student_answer}")

    try:
        question_index = int(question_index)
    except ValueError:
        print("Invalid question_index: not an integer")
        return jsonify({"error": "Question index must be an integer."}), 400

    try:
        session_doc = data_collection.find_one({"session_id": session_id})
        if not session_doc:
            print(f"No session found for session_id: {session_id}")
            return jsonify({"error": f"Invalid session ID: {session_id}"}), 400
    except Exception as e:
        print(f"Error querying session: {e}")
        return jsonify({"error": f"Database error: {e}"}), 500

    if question_index < 0 or question_index >= len(session_doc['questions']):
        print(f"Invalid question_index: {question_index}, questions length: {len(session_doc['questions'])}")
        return jsonify({"error": "Invalid question number."}), 400

    if not student_answer or len(student_answer.strip()) < 5:
        print("Student answer too short")
        return jsonify({"error": "Answer is too short."}), 400

    try:
        reference_answer = session_doc['questions'][question_index]['answer']
        evaluation = assistant.evaluate_answer(reference_answer, student_answer)

        response_doc = {
            "session_id": session_id,
            "question_index": question_index,
            "student_answer": student_answer,
            "evaluation": evaluation,
            "created_at": datetime.datetime.utcnow()
        }
        data_collection.insert_one(response_doc)

        return jsonify(evaluation)
    except Exception as e:
        print(f"Error evaluating answer: {e}")
        return jsonify({"error": f"Error evaluating answer: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)