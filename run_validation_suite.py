import requests
import time
import json
import sys

BASE_URL = "http://localhost:8001"

TEST_CASES = [
    {
        "name": "React Docs",
        "url": "https://react.dev/",
        "questions": [
            "What is React?",
            "Difference between Props and State",
            "Rules of Hooks",
            "What is React Compiler?",
            "What is React pricing?"
        ]
    },
    {
        "name": "Python Docs",
        "url": "https://docs.python.org/3/tutorial/appetite.html",
        "questions": [
            "What is Python?",
            "What is OOP?"
        ]
    },
    {
        "name": "FastAPI Docs",
        "url": "https://fastapi.tiangolo.com/",
        "questions": [
            "What is FastAPI?"
        ]
    },
    {
        "name": "GitHub Docs",
        "url": "https://github.com/about",
        "questions": [
            "What are GitHub Actions?"
        ]
    },
    {
        "name": "Flipkart",
        "url": "https://www.flipkart.com/",
        "questions": [
            "Toys"
        ]
    },
    {
        "name": "Real Madrid",
        "url": "https://www.realmadrid.com/en-US",
        "questions": [
            "Players"
        ]
    }
]

def run_suite():
    sys.stdout.reconfigure(encoding='utf-8')
    print("==================================================")
    print("        RAG X VALIDATION SUITE STARTING           ")
    print("==================================================")
    
    overall_success = True
    
    for case in TEST_CASES:
        name = case["name"]
        url = case["url"]
        print(f"\n[Test Case Group: {name}]")
        print(f"Indexing URL: {url}...")
        
        try:
            res = requests.post(f"{BASE_URL}/api/analyze", json={"url": url, "max_pages": 1}, timeout=30)
            res.raise_for_status()
            data = res.json()
            analysis_id = data["id"]
            print(f"[OK] Indexing Successful! stats: {data['kb_stats']}")
        except Exception as e:
            print(f"[ERROR] Indexing Failed for {url}: {e}")
            overall_success = False
            continue

        for q in case["questions"]:
            print(f"\n  Query: \"{q}\"")
            try:
                chat_res = requests.post(f"{BASE_URL}/api/chat", json={
                    "analysis_id": analysis_id,
                    "message": q,
                    "top_k": 3
                }, timeout=30)
                chat_res.raise_for_status()
                chat_data = chat_res.json()
                answer = chat_data["answer"]
                sources = chat_data.get("sources", [])
                
                print(f"  Response:\n{answer}\n")
                
                # Check for refusal responses
                is_refusal = any(ref in answer for ref in [
                    "Information not found", 
                    "unrelated to the indexed website",
                    "could not find pricing information",
                    "was not indexed during crawling"
                ])
                
                if is_refusal:
                    print("  Verification: [OK] Correct Refusal Response.")
                else:
                    # Check for citations
                    has_citations = "[Source" in answer or "Source Cards" in answer or "Source:" in answer or "source:" in answer.lower()
                    if has_citations:
                        print("  Verification: [OK] Response generated with citations.")
                    else:
                        print("  Verification: [ERROR] Response has no citations.")
                        overall_success = False
                        
                    # Check for natural language constraints (no "According to chunk", etc.)
                    banned_meta = ["according to chunk", "the following section", "this page contains", "based on the retrieved context"]
                    has_banned_meta = any(b in answer.lower() for b in banned_meta)
                    if has_banned_meta:
                        print("  Verification: [ERROR] Response contains banned meta-commentary.")
                        overall_success = False
                    else:
                        print("  Verification: [OK] Response language is direct and natural.")
                        
                print(f"  Sources: {[s.get('source_title', 'No Title') + ' - ' + s.get('source_url', '') for s in sources]}")
                print("-" * 50)
            except Exception as e:
                print(f"  [ERROR] Chat endpoint failed: {e}")
                overall_success = False
            
            # Simple rate limiting mitigation
            time.sleep(2)
            
    print("\n==================================================")
    if overall_success:
        print("          VALIDATION SUITE SUCCESSFUL!            ")
    else:
        print("          VALIDATION SUITE DETECTED FAILURES.     ")
    print("==================================================")
    
    if not overall_success:
        sys.exit(1)

if __name__ == "__main__":
    run_suite()
