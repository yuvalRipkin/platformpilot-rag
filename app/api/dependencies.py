from fastapi import Request

from app.services.answer_generator import AnswerGenerator
from app.services.embedder import Embedder
from app.services.retriever import Retriever


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def get_answer_generator(request: Request) -> AnswerGenerator:
    return request.app.state.answer_generator
