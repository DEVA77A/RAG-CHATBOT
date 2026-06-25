import requests
import time
import json

BASE_URL = "http://localhost:8001"

TEST_CASES = [
    {
        "url": "https://docs.python.org/3/tutorial/appetite.html",
        "questions": [
            "What is Python?",
            "What does Whetting Your Appetite explain?",
            "What is the Standard Library?",
            "Who won the FIFA World Cup?"
        ]
    },
    {
        "url": "https://huggingface.co/",
        "questions": [
            "What is Hugging Face?",
            "What products are offered?"
        ]
    },
    {
        "url": "https://github.com/about",
        "questions": [
            "What is GitHub?",
            "What features does GitHub provide?"
        ]
    },
    {
        "url": "https://fastapi.tiangolo.com/",
        "questions": [
            "What is FastAPI?",
            "How do I create my first API?"
        ]
    }
]

def run_validation():
    print("Starting Validation...")
    for case in TEST_CASES:
        url = case["url"]
        print(f"\n======================================")
        print(f"--- Indexing {url} ---")
        try:
            res = requests.post(f"{BASE_URL}/api/analyze", json={"url": url, "max_pages": 1})
            res.raise_for_status()
            data = res.json()
            analysis_id = data["id"]
            print(f"Index complete: {data['kb_stats']}")
        except Exception as e:
            print(f"Failed to index {url}: {e}")
            continue

        for q in case["questions"]:
            print(f"\nQ: {q}")
            try:
                chat_res = requests.post(f"{BASE_URL}/api/chat", json={
                    "analysis_id": analysis_id,
                    "message": q,
                    "top_k": 3
                })
                chat_res.raise_for_status()
                chat_data = chat_res.json()
                print(f"A: {chat_data['answer']}")
                sources = chat_data.get("sources", [])
                if sources:
                    print(f"Sources: {[s.get('title', 'No Title') + ' - ' + s.get('url', '') for s in sources]}")
                else:
                    print("No sources.")
            except Exception as e:
                print(f"Chat failed: {e}")
            
            time.sleep(3) # rate limit mitigation

if __name__ == "__main__":
    run_validation()
