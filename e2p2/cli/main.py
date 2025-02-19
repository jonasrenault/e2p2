import typer

from e2p2.layout.task import app as layout_app

app = typer.Typer(no_args_is_help=True)

app.add_typer(layout_app)


if __name__ == "__main__":
    app()
