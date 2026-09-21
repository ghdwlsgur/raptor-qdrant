from abc import ABC, abstractmethod


class BaseChatbotModel(ABC):
    @abstractmethod
    def answer(self, context: str, question: str) -> str:
        """컨텍스트를 근거로 질문에 답한다."""

    @property
    def describe(self) -> str:
        return self.__class__.__name__
