import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import typer

from e2p2.layout.task import app as layout_app
from e2p2.mfd.task import app as mfd_app
from e2p2.mfr.task import app as mfr_app
from e2p2.ocr.task import app as ocr_app
from e2p2.table.task import app as table_app

app = typer.Typer(no_args_is_help=True)

app.add_typer(layout_app)
app.add_typer(mfd_app)
app.add_typer(mfr_app)
app.add_typer(ocr_app)
app.add_typer(table_app)


if __name__ == "__main__":
    app()
