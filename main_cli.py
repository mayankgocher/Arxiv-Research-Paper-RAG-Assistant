import argparse
import sys
from src.agent import run_query
from src.config import Config

def main():
    parser = argparse.ArgumentParser(description="Arxiv Research Paper Assistant CLI")
    parser.add_argument("--query", "-q", type=str, help="The question to ask the agent.", required=False)
    args = parser.parse_args()

    if not Config.OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable is not set. Please check your .env file.")
        sys.exit(1)

    if args.query:
        answer, context = run_query(args.query)
        print("\n--- Answer ---")
        print(answer)
        print("\n--- Sources Used ---")
        for i, doc in enumerate(context):
            source = doc.metadata.get("source", "Unknown")
            print(f"[{i+1}] Source: {source}")
    else:
        print("Welcome to the Arxiv Research Paper Assistant!")
        print("Type 'exit' or 'quit' to exit.")
        while True:
            try:
                user_input = input("\nAsk a question: ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                if not user_input.strip():
                    continue
                
                answer, context = run_query(user_input)
                print("\n--- Answer ---")
                print(answer)
                print("\n--- Sources Used ---")
                # Deduplicate sources for cleaner output
                sources = list(set([doc.metadata.get("source", "Unknown") for doc in context]))
                for i, source in enumerate(sources):
                    print(f"[{i+1}] {source}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
