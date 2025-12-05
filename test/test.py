import pandas as pd
import httpx
import asyncio
from typing import Dict
import time

OPENROUTER_API_KEY = ""  # ключ
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODELS = [
    "google/gemini-2.5-flash-preview-09-2025",
    "microsoft/phi-4",
    "anthropic/claude-3.5-sonnet",
    "mistralai/ministral-3b-2512",
    "qwen/qwen3-4b:free",
    "meta-llama/llama-3.1-8b-instruct",
    "deepseek/deepseek-v3.2-speciale"
]

DATA_CONTEXT = """Доступные данные из сводной таблицы:
                
                Всего записей: 921,612
                Уникальных пациентов: 847,670
                Уникальных диагнозов: 2,433
                Уникальных препаратов: 2,072
                Записей с ценой: 671,087
                Средняя цена препарата: 4022.44 руб.
                
                Структура данных включает:
                - Информацию о пациентах (пол, возраст, регион)
                - Диагнозы с классификацией МКБ-10
                - Препараты с дозировками и ценами
                - Даты назначений для анализа сезонности"""


async def call_openrouter(model: str, question: str, answer: str, text: str) -> Dict:
    prompt = f"""Контекст:
                {DATA_CONTEXT}
                
                Данные анализа:
                
                Q: {question}
                A: {answer}
                
                Информация: {text}
                
                Вопрос: {question}
                
                Проанализируй данные и дай четкий структурированный ответ с конкретными цифрами."""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Refactor": "test-medical-agent",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 512,
        "temperature": 0.3
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )

            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response": result["choices"][0]["message"]["content"],
                    "model": model
                }
            else:
                return {
                    "success": False,
                    "error": f"Status {response.status_code}: {response.text}",
                    "model": model
                }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model": model
        }


def extract_expected_answer(answer_text: str) -> str:
    return answer_text.strip().split('\n')[0].lower()


def check_answer_correctness(model_response: str, expected_answer: str) -> bool:
    response_lower = model_response.lower()

    if expected_answer in ["да", "yes"]:
        return any(word in response_lower for word in
                   ["да", "yes", "да.", "yes.", "верно", "правда", "согласен", "подтверждаю"])
    elif expected_answer in ["нет", "no"]:
        return any(word in response_lower for word in
                   ["нет", "no", "нет.", "no.", "неверно", "ложь", "не согласен", "отрицаю"])

    keywords = expected_answer.split()
    matches = sum(1 for kw in keywords if kw in response_lower)
    return matches / len(keywords) >= 0.5 if keywords else False


async def evaluate_models(csv_path: str, num_samples: int = 100):
    df = pd.read_csv(csv_path)
    df = df.head(num_samples)

    print(f"Загружено {len(df)} вопросов")
    print(f"Тестирование {len(MODELS)} моделей...\n")

    results = {
        "model": [],
        "accuracy": [],
        "success_rate": [],
        "avg_response_time": [],
        "total_correct": [],
        "total_processed": []
    }

    for model in MODELS:
        print(f"\nТестирование модели: {model}")

        correct_count = 0
        success_count = 0
        response_times = []

        for idx, row in df.iterrows():
            question = row['question']
            expected_answer = extract_expected_answer(row['answer'])
            text = row.get('text', '')

            start_time = time.time()
            response = await call_openrouter(model, question, row['answer'], text)
            elapsed = time.time() - start_time
            response_times.append(elapsed)

            if response["success"]:
                success_count += 1
                if check_answer_correctness(response["response"], expected_answer):
                    correct_count += 1
                print(f"  [{idx + 1}/{len(df)}] ✓ Correct" if check_answer_correctness(response["response"],
                                                                                       expected_answer) else f"  [{idx + 1}/{len(df)}] ✗ Wrong")
            else:
                print(f"  [{idx + 1}/{len(df)}] ERROR: {response['error']}")

            await asyncio.sleep(0.5)

        accuracy = (correct_count / success_count * 100) if success_count > 0 else 0
        success_rate = (success_count / len(df) * 100)
        avg_time = sum(response_times) / len(response_times) if response_times else 0

        results["model"].append(model)
        results["accuracy"].append(accuracy)
        results["success_rate"].append(success_rate)
        results["avg_response_time"].append(avg_time)
        results["total_correct"].append(correct_count)
        results["total_processed"].append(len(df))

        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Success rate: {success_rate:.2f}%")
        print(f"  Avg response time: {avg_time:.2f}s")

    results_df = pd.DataFrame(results)
    results_df.to_csv("model_evaluation_results.csv", index=False)

    print(results_df.to_string(index=False))
    print("\nРезультаты сохранены в 'model_evaluation_results.csv'")

    return results_df


async def main():
    csv_path = "medical_training_data.csv"
    results = await evaluate_models(csv_path, num_samples=100)


if __name__ == "__main__":
    asyncio.run(main())
