import typer

from e2p2.formula_detection.task import app as mfd_app
from e2p2.formula_recognition.task import app as mfr_app
from e2p2.layout.task import app as layout_app

app = typer.Typer(no_args_is_help=True)

app.add_typer(layout_app)
app.add_typer(mfd_app)
app.add_typer(mfr_app)


if __name__ == "__main__":
    app()
