from enum import Enum

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

from pydantic import BaseModel
from pydantic.typing import List

from sadedegel.bblock import Doc, Sentences
from sadedegel.summarize import RandomSummarizer, PositionSummarizer, Rouge1Summarizer
from sadedegel.about import __version__

import click
import numpy as np

from loguru import logger


class TimeUnitEnum(str, Enum):
    """Time unit is used to specify duration reader wants to spend in reading.
    Together with wpm defines the summary length to be returned.
    """
    SECOND = 'second'
    MINUTE = 'minute'


class BasicRequest(BaseModel):
    """Summarization request."""
    doc: str
    wpm: int = 170


class Request(BasicRequest):
    """Summarization request with duration added."""
    duration: int = 5
    unit: TimeUnitEnum = TimeUnitEnum.MINUTE


class DocSummary(BaseModel):
    """Document statistics summary.
    Used by summarization service calculate front end material.
    """
    sentence_count: int
    word_count: int


class RichSummary(DocSummary):
    """Enhanced statistics summary with reading duration."""
    wpm: int
    duration: float
    unit: TimeUnitEnum


class Response(BaseModel):
    sentences: List[str]
    original: DocSummary
    summary: DocSummary


class APIInfo(BaseModel):
    """API Info including
    Version installed
    Maintainers email
    sadedeGel website
    Current pyPI version available
    """
    version: str
    email: str
    website: str
    github: str


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
        "url": "https://sadedegel.ai/dist/img/logo-2.png?s=280&v=4"
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


def summary_filter(sents, scores, word_count, limit=None):
    if not (len(sents) == len(scores) and len(scores) == len(word_count)):
        raise ValueError(
            f"Length of sents ({len(sents)}), scores ({len(scores)}) and word_count ({len(word_count)}) should match.")

    rank = np.argsort(scores)[::-1]

    if limit:
        selected_sents_idx = rank[word_count[rank].cumsum() <= limit]
    else:
        selected_sents_idx = rank

    selected_sents_idx.sort()  # sort in ascending order to preserve order

    return [sents[i] for i in selected_sents_idx]


@app.get("/", include_in_schema=False)
async def home():
    return RedirectResponse("http://sadedegel.ai")


@app.get("/api/info", tags=["Information"], summary="sadedeGel metadata")
async def info():
    return APIInfo(version=__version__, email="info@sadedegel.ai", website="sadedegel.ai",
                   github="https://github.com/GlobalMaksimum/sadedegel")


def summarize(summarizer, sentences: List[Sentences], limit: float) -> Response:
    word_count = np.array([len(s) for s in sentences], dtype=np.int)

    scores = summarizer.predict(sentences)

    logger.info(scores)
    logger.info(word_count)

    sentences_limited = summary_filter(sentences, scores, word_count, limit)

    logger.info(sentences_limited)

    return Response(sentences=[sentences_limited[i].text for i in range(len(sentences_limited))],
                    original=DocSummary(sentence_count=len(sentences), word_count=word_count.sum()),
                    summary=DocSummary(sentence_count=len(sentences_limited),
                                       word_count=np.array([len(s) for s in sentences_limited],
                                                           dtype=np.int).sum()))


@app.post("/api/doc/statistics", tags=["Utility"], summary="Calculate approximate reading duration of a document")
async def duration(req: BasicRequest):
    """Calculates the approximate reading duration for the document based on reader read speed"""

    sentences = list(Doc(req.doc))

    word_count = sum((len(s) for s in sentences))

    dur = (word_count / (req.wpm / 60))
    unit = TimeUnitEnum.SECOND

    logger.info(f"Total duration {dur}")

    return RichSummary(sentence_count=len(sentences), word_count=word_count, wpm=req.wpm, duration=dur, unit=unit)


@app.post("/api/summarizer/random", tags=["Summarizer"], summary="Use RandomSummarizer")
async def random(req: Request):
    """Baseline [Random Summarizer](https://github.com/GlobalMaksimum/sadedegel/tree/master/sadedegel/summarize/README.md)

            Picks up random sentences until total number of tokens is less than equal to `wpm x duration`
    """

    sentences = list(Doc(req.doc))
    logger.info(sentences)

    if req.unit == TimeUnitEnum.MINUTE:
        duration_in_min = req.duration
    else:
        duration_in_min = req.duration / 60.

    return summarize(RandomSummarizer(seed=None), sentences, req.wpm * duration_in_min)


@app.post("/api/summarizer/firstk", tags=["Summarizer"],
          summary="Use Position Summarizer to obtain a FirstK summarizer")
async def firstk(req: Request):
    """Baseline [Position Summarizer](https://github.com/GlobalMaksimum/sadedegel/tree/master/sadedegel/summarize/README.md)

            Picks up first a few sentences until total number of tokens is less than equal to `wpm x duration`
    """

    sentences = list(Doc(req.doc))

    if req.unit == TimeUnitEnum.MINUTE:
        duration_in_min = req.duration
    else:
        duration_in_min = req.duration / 60.

    return summarize(PositionSummarizer(), sentences, req.wpm * duration_in_min)


@app.post("/api/summarizer/rouge1", tags=["Summarizer"],
          summary="Use unsupervised Rouge1 Summarizer")
async def rouge1(req: Request):
    """Baseline [Rouge1 Summarizer](https://github.com/GlobalMaksimum/sadedegel/tree/master/sadedegel/summarize/README.md)

            Rank sentences based on their rouge1 score in Document and return a list of sentences until number of total tokens is less than equal to `wpm x duration`
    """

    sentences = list(Doc(req.doc))

    if req.unit == TimeUnitEnum.MINUTE:
        duration_in_min = req.duration
    else:
        duration_in_min = req.duration / 60.

    return summarize(Rouge1Summarizer(), sentences, req.wpm * duration_in_min)


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
