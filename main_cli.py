import argparse
import sys
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.text import Text
from src.agent import run_query
from src.config import Config

console = Console()

def display_response(answer: str, context: list):
    console.print("\n")
    # Using Markdown for the answer allows nice formatting if the LLM returns markdown
    console.print(Panel(Markdown(answer), title="[bold green]Assistant Answer[/bold green]", border_style="green"))
    
    if context:
        sources_text = Text()
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in context]))
        for i, source in enumerate(sources):
            sources_text.append(f"[{i+1}] {source}\n")
        
        console.print(Panel(sources_text, title="[bold blue]Sources Used[/bold blue]", border_style="blue"))
    console.print("\n")

def main():
    parser = argparse.ArgumentParser(description="Arxiv Research Paper Assistant CLI")
    parser.add_argument("--query", "-q", type=str, help="The question to ask the agent.", required=False)
    args = parser.parse_args()

    if not Config.OPENAI_API_KEY:
        console.print("[bold red]Error:[/bold red] OPENAI_API_KEY environment variable is not set. Please check your .env file.")
        sys.exit(1)

    if args.query:
        with console.status("[bold cyan]Searching and generating answer...[/bold cyan]", spinner="dots"):
            answer, context = run_query(args.query)
        display_response(answer, context)
    else:
        console.print(Panel.fit("[bold magenta]Welcome to the Arxiv Research Paper Assistant![/bold magenta]\nType 'exit' or 'quit' to exit.", border_style="magenta"))
        
        while True:
            try:
                user_input = Prompt.ask("[bold yellow]Ask a question[/bold yellow]")
                if user_input.lower() in ['exit', 'quit']:
                    console.print("[bold magenta]Goodbye![/bold magenta]")
                    break
                if not user_input.strip():
                    continue
                
                with console.status("[bold cyan]Searching and generating answer...[/bold cyan]", spinner="dots"):
                    answer, context = run_query(user_input)
                
                display_response(answer, context)
            except KeyboardInterrupt:
                console.print("\n[bold magenta]Goodbye![/bold magenta]")
                break
            except Exception as e:
                console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")

if __name__ == "__main__":
    main()
