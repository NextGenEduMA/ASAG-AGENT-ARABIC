# Install necessary libraries


import os
import re
import time
import google.generativeai as genai
import numpy as np
from sentence_transformers import SentenceTransformer, util

API_KEY = input("Enter your Google API Key: ")

if not API_KEY:
    print("ERROR: No API key provided. You need a Google API key to use Gemini models.")
else:
    try:
        genai.configure(api_key=API_KEY)
        # Use Gemini 1.5 Flash for efficient responses
        model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
        print("Gemini model initialized successfully.")
    except Exception as e:
        print(f"ERROR initializing Gemini model: {e}")
        model = None

class ArabicEducationalAssistant:
    def __init__(self, genai_model):
        self.model = genai_model
        # Initialize sentence transformer for semantic evaluation
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        print("Educational Assistant initialized successfully!")

    def generate_questions(self, content, num_questions=5):
        """Generate open-ended questions based on the provided content"""
        if not content or len(content.strip()) < 50:
            return [{"question": "يرجى تقديم نص أطول لتوليد أسئلة ذات معنى.", "answer": "النص المقدم قصير جدًا."}]

        prompt = f"""
        أنت مساعد تعليمي متخصص في اللغة العربية. مهمتك هي قراءة النص التالي وإنشاء {num_questions} أسئلة **مفتوحة** مع إجاباتها النموذجية.

        يجب أن تتطلب هذه الأسئلة تفكيرًا وفهمًا للنص، وليس مجرد استرجاع مباشر للمعلومات.
        ركز على الأسباب، النتائج، المقارنات، أو التطبيقات المحتملة المذكورة أو التي يمكن استنتاجها من النص.
        تجنب الأسئلة التي يمكن الإجابة عليها بـ "نعم" أو "لا" أو بكلمة واحدة.

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

            # Parse the response to extract question-answer pairs
            qa_pairs = self.extract_qa_pairs(response.text)
            return qa_pairs

        except Exception as e:
            print(f"Error generating questions: {e}")
            return []

    def extract_qa_pairs(self, text):
        """Extract question-answer pairs from generated text"""
        qa_pairs = []

        # Split text into lines for processing
        lines = text.split('\n')

        current_q = ""
        current_a = ""
        qa_state = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect question patterns
            if re.match(r'^\d+[\.\)]', line) or "سؤال" in line.lower() or ":" in line and not current_q:
                if current_q and current_a:
                    qa_pairs.append({"question": current_q, "answer": current_a})
                current_q = line
                current_a = ""
                qa_state = "question"
            # Detect answer patterns
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

        # Add the last pair
        if current_q and current_a:
            qa_pairs.append({"question": current_q, "answer": current_a})

        # If failed to extract in the expected format, try a simpler approach
        if not qa_pairs:
            # Look for numbered questions
            questions = []
            answers = []

            # Try to find numbered patterns
            q_blocks = re.split(r'\d+[\.\)]', text)
            if len(q_blocks) > 1:
                # First element is likely introduction text, skip it
                q_blocks = q_blocks[1:]

                for block in q_blocks:
                    parts = re.split(r'الإجابة|إجابة|الجواب|:', block, 1)
                    if len(parts) > 1:
                        questions.append(parts[0].strip())
                        answers.append(parts[1].strip())
                    else:
                        # If no clear answer separator, just add the entire block as a question
                        questions.append(block.strip())
                        answers.append("يرجى الرجوع إلى النص لتحديد الإجابة المناسبة.")

                # Create pairs from what we found
                for i in range(len(questions)):
                    qa_pairs.append({"question": questions[i], "answer": answers[i] if i < len(answers) else ""})

        return qa_pairs

    def evaluate_answer(self, reference_answer, student_answer):
        """Evaluate student's answer against reference answer"""
        if not student_answer or len(student_answer.strip()) < 5:
            return {
                "score": 1,
                "feedback": "الإجابة قصيرة جدًا أو غير مكتملة.",
                "detailed_feedback": "يرجى تقديم إجابة أكثر تفصيلاً."
            }

        # First, use semantic similarity for an initial score
        reference_embedding = self.sentence_model.encode(reference_answer, convert_to_tensor=True)
        student_embedding = self.sentence_model.encode(student_answer, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(reference_embedding, student_embedding).item()

        # Use Gemini for detailed evaluation
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

        قم بتقييم إجابة الطالب على مقياس من 1 إلى 5 نقاط، حيث:
        1 = غير مقبول
        2 = ضعيف
        3 = متوسط
        4 = جيد
        5 = ممتاز

        قدم النتيجة في السطر الأول بالصيغة التالية: "النتيجة: X/5"

        ثم قدم تقييمًا مفصلاً موجزًا (3-5 أسطر) يشرح نقاط القوة والضعف في إجابة الطالب.
        كن إيجابيًا ومشجعًا وقدم اقتراحات محددة للتحسين.
        """

        try:
            response = self.model.generate_content(prompt)
            if not response or not hasattr(response, 'text') or not response.text:
                # Fallback to similarity score if Gemini fails
                return self.generate_fallback_evaluation(similarity)

            # Parse the response
            evaluation_text = response.text.strip()

            # Extract the score
            score_match = re.search(r'النتيجة:\s*(\d)[/\.]5', evaluation_text)
            score = int(score_match.group(1)) if score_match else round(similarity * 5)

            # Extract the feedback
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
        """Generate a fallback evaluation based on similarity score"""
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

def main():
    if 'model' not in globals() or model is None:
        print("ERROR: Gemini model not initialized. Please check your API key.")
        return

    # Initialize the assistant
    print("Initializing Arabic Educational Assistant...")
    assistant = ArabicEducationalAssistant(model)

    # Text input options
    print("\nHow would you like to provide the educational content?")
    print("1. Enter text directly")
    print("2. Load from a text file")
    
    choice = input("Enter your choice (1 or 2): ")
    
    content = ""
    if choice == "1":
        print("\nEnter your educational text (type 'END' on a new line when finished):")
        while True:
            line = input()
            if line.strip() == "END":
                break
            content += line + "\n"
    elif choice == "2":
        # Remplacer files.upload() par une entrée de chemin de fichier standard
        file_path = input("\nEnter the path to your text file: ")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"Successfully loaded file: {file_path}")
        except UnicodeDecodeError:
            # Essayer un encodage alternatif
            try:
                with open(file_path, 'r', encoding='cp1256') as f:
                    content = f.read()
                print(f"Successfully loaded file with cp1256 encoding: {file_path}")
            except Exception as e:
                print(f"Error reading file: {e}")
                return
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    else:
        print("Invalid choice. Exiting.")
        return

    # Check if we have content
    if not content or len(content.strip()) < 50:
        print("Insufficient content provided. Please provide more text.")
        return

    # Generate questions
    print("\nGenerating questions...")
    questions = assistant.generate_questions(content)

    if not questions:
        print("Failed to generate questions. Please try again with different content.")
        return

    # Display generated questions
    print("\n=== Generated Questions and Answers ===")
    for i, qa in enumerate(questions):
        print(f"\nQuestion {i+1}: {qa['question']}")
        print(f"Model Answer: {qa['answer']}")

    # Save questions to a file
    with open('generated_questions.txt', 'w', encoding='utf-8') as f:
        for i, qa in enumerate(questions):
            f.write(f"Question {i+1}: {qa['question']}\n")
            f.write(f"Model Answer: {qa['answer']}\n\n")
    print("\nQuestions saved to 'generated_questions.txt'")

    # Interactive student evaluation
    print("\n=== Student Evaluation Mode ===")
    print("You can test the system by answering questions and receiving feedback.")

    while True:
        try:
            q_num = int(input("\nEnter question number to answer (or 0 to exit): "))
            if q_num == 0:
                break

            if 1 <= q_num <= len(questions):
                print(f"\nQuestion: {questions[q_num-1]['question']}")
                student_answer = input("Your answer: ")

                print("\nEvaluating your answer...")
                evaluation = assistant.evaluate_answer(questions[q_num-1]['answer'], student_answer)

                # Display evaluation results
                print(f"\nScore: {evaluation['score']}/5")
                print(f"Feedback: {evaluation['feedback']}")
                print(f"Detailed Feedback: {evaluation['detailed_feedback']}")
            else:
                print(f"Invalid question number. Please enter a number between 1 and {len(questions)}.")
        except ValueError:
            print("Please enter a valid number.")

    print("\nThank you !")

if __name__ == "__main__":
    main()