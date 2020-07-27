from fastapi.openapi.utils import get_openapi

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic.typing import List
from sadedegel.tokenize import Doc
from math import ceil
from sadedegel.summarize import RandomSummarizer, PositionSummarizer
import click
import numpy as np
from sadedegel.about import __version__


class Document(BaseModel):
    text: str


class Summary(BaseModel):
    sentences: List[str]
    length: int
    ratio: float
    summary_length: int


app = FastAPI()


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="sadedeGel",
        version=__version__,
        description="sadedeGel is an extraction-based summarizer for Turkish news content",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://avatars0.githubusercontent.com/u/2204565?s=280&v=2"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

app.add_middleware(
    CORSMiddleware,
    allow_origins='*',
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/sadedegel/random")
async def random(doc: Document, ratio: float = 0.2):
    """Baseline Random summarizer.
        Returning random K sentence of the document.
    """
    sentences = Doc(doc.text).sents

    summary_length = max(1, ceil(len(sentences) * ratio))

    summary = RandomSummarizer().predict(sentences)
    top_index = np.argsort(summary)[::-1][:summary_length]

    return Summary(sentences=[s.text for i, s in enumerate(sentences) if i in top_index], length=len(sentences),
                   summary_length=summary_length,
                   ratio=ratio)


@app.post("/sadedegel/firstk")
async def firstk(doc: Document, ratio: float = 0.2):
    """Baseline FirstK summarizer.
    Returning first K sentence of the document
    """
    sentences = Doc(doc.text).sents

    summary_length = max(1, ceil(len(sentences) * ratio))

    summary = PositionSummarizer().predict(sentences)
    top_index = np.argsort(summary)[::-1][:summary_length]

    return Summary(sentences=[s.text for i, s in enumerate(sentences) if i in top_index], length=len(sentences),
                   summary_length=summary_length,
                   ratio=ratio)


@click.command()
@click.option("--host", '-h', help="Hostname", default="0.0.0.0")  # nosec
@click.option("--log-level",
              type=click.Choice(['debug', 'info'], case_sensitive=False), help="Logging Level", default="info")
@click.option("--reload", is_flag=True, default=False, help="enable/disable auto reload for development.")
@click.option("--port", help="Port", default=8000)
def server(host, log_level, reload, port):
    """Span a sadedeGel http server instance."""

    if reload:
        uvicorn.run("sadedegel.server.__main__:app", host=host, log_level=log_level, reload=reload, port=port)
    else:
        uvicorn.run(app, host=host, log_level=log_level, reload=reload, port=port)


if __name__ == "__main__":
    server()
