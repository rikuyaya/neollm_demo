from neollm import MyLLM
from dotenv import load_dotenv

from neollm.types import Messages, Response


class Test_LLM(MyLLM):

    def _preprocess(self, inputs: dict[str, str]) -> Messages:
        system_prompt = (
            "挨拶して"
        )
        user_prompt = (
            "こんにちは"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return messages
    
    # def _ruleprocess(self):
    #     return None

    # def _update_settings(self):
    #     pass

    def _postprocess(self, response: Response) -> str:
        return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    # .envの読み込み
    load_dotenv()

    testllm = Test_LLM(
        platform="openai",
        model="gpt-4o-mini-2024-07-18",
        verbose=False,
        llm_settings={
            "temperature": 0.5,
            "max_tokens": 1024,
        },
    )

    output = testllm(inputs={})
    print(output)
    
