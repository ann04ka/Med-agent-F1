import os
import requests

class GigaChatConfig:

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "") # ключ
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    MODEL_NAME = "qwen/qwen3-vl-30b-a3b-instruct"

    TEMPERATURE = 0.3
    MAX_TOKENS = 2048
    TOP_P = 0.9

    SYSTEM_PROMPT = """Ты - профессиональный медицинский аналитик и эксперт по обработке данных.
                    
                    ТВОЯ ЗАДАЧА:
                    - Анализировать медицинские данные Санкт-Петербурга
                    - Отвечать четко, структурировано и по существу
                    - Использовать конкретные цифры и статистику из данных
                    - Предоставлять практические рекомендации
                    
                    ФОРМАТ ОТВЕТА:
                    - Начинай сразу с ответа, БЕЗ вводных фраз типа "Конечно", "Хорошо", "Давайте разберем"
                    - Структурируй информацию с помощью заголовков и списков
                    - Указывай точные цифры и проценты
                    - При рекомендации препаратов указывай цены и частоту назначения
                    - Если данных недостаточно, честно об этом сообщай
                    
                    ЗАПРЕЩЕНО:
                    - Придумывать данные или статистику
                    - Использовать вводные и вежливые фразы
                    - Давать расплывчатые ответы
                    - Упоминать о своих ограничениях
                    
                    Отвечай четко, профессионально и информативно."""

    @staticmethod
    def create_pipeline():
        if not GigaChatConfig.OPENROUTER_API_KEY:
            print("WARNING: OPENROUTER_API_KEY not set")

        return {
            "api_key": GigaChatConfig.OPENROUTER_API_KEY,
            "base_url": GigaChatConfig.OPENROUTER_BASE_URL,
            "model": GigaChatConfig.MODEL_NAME
        }

    @staticmethod
    def generate_text(pipeline: dict, prompt: str, max_length: int = 2048) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {pipeline['api_key']}",
                "Content-Type": "application/json"
            }

            data = {
                "model": pipeline['model'],
                "messages": [
                    {
                        "role": "system",
                        "content": GigaChatConfig.SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": GigaChatConfig.TEMPERATURE,
                "max_tokens": max_length,
                "top_p": GigaChatConfig.TOP_P
            }

            response = requests.post(
                pipeline['base_url'],
                headers=headers,
                json=data,
                timeout=60
            )

            response.raise_for_status()
            result = response.json()

            return result['choices'][0]['message']['content']

        except Exception as e:
            print(f"API Error: {e}")
            raise


if __name__ == "__main__":
    pipe = GigaChatConfig.create_pipeline()
    result = GigaChatConfig.generate_text(pipe, "Привет, сколько месяцев в году?", max_length=100)
    print(result)
