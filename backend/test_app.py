import requests
import json
import os

def load_question(filenames) :
    questions = []
    
    for filename in filenames:
        with open(filename, 'r') as file:
            data = json.load(file)
            questions.extend(data["questions"])
    return questions

def save_responses(answers, filename = "responses.json"):
    with open(filename, 'w') as file:
        json.dump({"responses" : answers}, file, indent=4)

# Tesing function
def test_questions() :
    # list of question files
    question_files = [
        'YesNoQuestions.json',
        'ExplanationQuestions.json',
        'ListQuestions.json',
        'ComparisonQuestions.json',
        'OtherQuestions.json'
    ]
 
    questions = load_question(question_files)
    answers = []

    for item in questions:
        current_question = item["question"]
        print(f"Testing question: {current_question}")
        # Send a POST request to the API
        response = requests.post("http://127.0.0.1:5000/api/chat/test_questions", json={"content": current_question})

        if response.status_code == 200:
            answer = response.json()
            answers.append({
                "question": current_question,
                "response": answer["content"]
            })
            print(f"Response: {answer["content"]}")
        else:
            print(f"Error: {response.status_code} - {response.text}")

    # Save all responses to the JSON file
    save_responses(answers)

    # Check if all respoinses.json file was created
    if os.path.exists('responses.json'):
        print("Responses saved to respinses.json.")
    else:
        print("Failed to save responses.")
if __name__ == "__main__":
    test_questions()



