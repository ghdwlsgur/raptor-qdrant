from abc import ABC, abstractmethod

from raptor_qdrant.rag.utils import load_prompt

CHATBOT_PROMPT = "prompt/chatbot.md"


class BaseChatbotModel(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str:
        """프롬프트를 그대로 모델에 넘기고 답을 받는다."""

    def answer(self, context: str, question: str) -> str:
        """컨텍스트를 근거로 질문에 답한다.

        요약처럼 질의응답이 아닌 호출은 complete 를 직접 쓴다. 그것까지 답변
        템플릿에 감싸면 "컨텍스트를 근거로 답하라"는 지시가 요약 프롬프트를
        질문 자리에 끼워 넣은 꼴이 된다.
        """
        return self.complete(
            load_prompt(CHATBOT_PROMPT).format(
                context=context, question=question
            )
        )

    @property
    def describe(self) -> str:
        return self.__class__.__name__
