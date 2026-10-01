"""
Interactive CLI QA Server
加载已有索引，直接问答
"""

from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

PROJECT_ROOT = Path(__file__).resolve().parent

from pipeline import RAGPipeline, load_config
from src.generator import format_answer_for_display

console = Console()


def main():
    console.print(Panel.fit(
        "[bold blue]📚 arXiv Paper RAG QA[/bold blue]\n"
        "Type your question and press Enter.\n"
        "Commands: [bold]quit[/bold] | [bold]stats[/bold] | [bold]dense[/bold] / [bold]hybrid[/bold] (switch mode)",
        title="Welcome",
    ))

    config = load_config()
    pipeline = RAGPipeline(config)

    console.print("[yellow]Loading models...[/yellow]")
    pipeline.init_models(skip_llm=False)
    pipeline.load_existing_index()
    console.print("[green]✅ Ready![/green]\n")

    use_rerank = True

    while True:
        try:
            question = console.input("[bold cyan]❓ Question:[/bold cyan] ").strip()

            if question.lower() in ("quit", "exit", "q"):
                break
            elif question.lower() == "stats":
                faiss_count = pipeline.faiss_indexer.count()
                bm25_count = len(pipeline.bm25_index.chunks)
                console.print(f"[dim]FAISS vectors: {faiss_count}[/dim]")
                console.print(f"[dim]BM25 docs: {bm25_count}[/dim]")
                console.print(f"[dim]Rerank: {'ON' if use_rerank else 'OFF'}[/dim]")
                continue
            elif question.lower() == "dense":
                use_rerank = False
                console.print("[dim]Switched to dense-only mode (no rerank)[/dim]")
                continue
            elif question.lower() == "hybrid":
                use_rerank = True
                console.print("[dim]Switched to hybrid + rerank mode[/dim]")
                continue
            elif not question:
                continue

            result = pipeline.query(question, use_rerank=use_rerank)

            console.print()
            console.print(Panel(
                Markdown(result["answer"]),
                title="[bold green]Answer[/bold green]",
                border_style="green",
            ))

            if result.get("citations"):
                refs = []
                for c in result["citations"]:
                    refs.append(
                        f"[{c['index']}] {c['title']}\n"
                        f"    https://arxiv.org/abs/{c['arxiv_id']}"
                    )
                console.print(Panel(
                    "\n\n".join(refs),
                    title="[bold blue]📚 References[/bold blue]",
                    border_style="blue",
                ))
            console.print()

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    console.print("[dim]Bye! 👋[/dim]")


if __name__ == "__main__":
    main()
