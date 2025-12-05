from typing import Dict, Any
from gigachat_config import GigaChatConfig
from data_processor import DataProcessor
import pandas as pd


class MedicalAgentGigaChat:
    def __init__(self, data_processor: DataProcessor, use_cache: bool = True):
        self.processor = data_processor
        self.use_cache = use_cache

        print("Initializing medical agent...")

        self.pipeline = self._init_pipeline()
        self.data_context = self._prepare_data_context()

        print("Agent ready")

    def _init_pipeline(self):
        print("Loading AI model...")

        try:
            pipe = GigaChatConfig.create_pipeline()
            print("Model loaded")
            return pipe

        except Exception as e:
            print(f"Error: {e}")
            raise

    def _prepare_data_context(self) -> str:
        if self.processor.merged_data is None:
            return "Данные не загружены"

        merged_summary = self.processor.get_merged_data_summary()

        context = f"""Доступные данные из сводной таблицы:
                    
                    Всего записей: {merged_summary['total_records']:,}
                    Уникальных пациентов: {merged_summary['unique_patients']:,}
                    Уникальных диагнозов: {merged_summary['unique_diagnoses']:,}
                    Уникальных препаратов: {merged_summary['unique_drugs']:,}
                    Записей с ценой: {merged_summary['records_with_price']:,}
                    Средняя цена препарата: {merged_summary['avg_cost']:.2f} руб.
                    
                    Структура данных включает:
                    - Информацию о пациентах (пол, возраст, регион)
                    - Диагнозы с классификацией МКБ-10
                    - Препараты с дозировками и ценами
                    - Даты назначений для анализа сезонности
                    """
        print(context)
        return context.strip()

    def _analyze_from_merged(self, query_lower: str) -> str:
        if self.processor.merged_data is None:
            return "Сводная таблица не создана"

        df = self.processor.merged_data

        analysis = "Анализ на основе сводной таблицы:\n\n"

        analysis += f"Всего записей: {len(df):,}\n"
        analysis += f"Период данных: {df['дата_рецепта'].min()} - {df['дата_рецепта'].max()}\n"

        if 'пол' in df.columns:
            gender_dist = df['пол'].value_counts()
            analysis += f"\nРаспределение по полу:\n"
            for gender, count in gender_dist.items():
                analysis += f"  {gender}: {count:,} ({count / len(df) * 100:.1f}%)\n"

        if 'возрастная_группа' in df.columns:
            age_dist = df['возрастная_группа'].value_counts().head(5)
            analysis += f"\nТОП-5 возрастных групп:\n"
            for age, count in age_dist.items():
                analysis += f"  {age}: {count:,}\n"

        if 'название_диагноза' in df.columns:
            top_diseases = df['название_диагноза'].value_counts().head(5)
            analysis += f"\nТОП-5 диагнозов:\n"
            for disease, count in top_diseases.items():
                analysis += f"  {disease}: {count:,}\n"

        if 'сезон' in df.columns:
            season_dist = df['сезон'].value_counts()
            analysis += f"\nРаспределение по сезонам:\n"
            for season, count in season_dist.items():
                analysis += f"  {season}: {count:,}\n"

        return analysis.strip()

    def _gender_disease_comparison(self) -> str:
        comparison = self.processor.compare_diseases_by_gender()

        if comparison is None or len(comparison) == 0:
            return "Недостаточно данных для сравнения по полу"

        top_female = comparison.nlargest(10, 'Разница (Ж-М)')
        top_male = comparison.nsmallest(10, 'Разница (Ж-М)')

        analysis = "Сравнение заболеваний по полу\n\n"

        analysis += "Чаще у женщин:\n"
        for idx, row in top_female.iterrows():
            analysis += f"- {row['Название диагноза']}: {row['% женщин']:.1f}% женщин vs {row['% мужчин']:.1f}% мужчин\n"

        analysis += "\nЧаще у мужчин:\n"
        for idx, row in top_male.iterrows():
            analysis += f"- {row['Название диагноза']}: {row['% мужчин']:.1f}% мужчин vs {row['% женщин']:.1f}% женщин\n"

        return analysis.strip()

    def _drug_recommendations(self, diagnosis_keyword: str) -> str:
        drugs = self.processor.get_drug_recommendations(diagnosis_keyword, top_n=15)

        if drugs is None or len(drugs) == 0:
            return f"Препараты для '{diagnosis_keyword}' не найдены в базе данных"

        analysis = f"Препараты для лечения '{diagnosis_keyword}'\n\n"

        for idx, row in drugs.iterrows():
            analysis += f"{idx + 1}. {row['Препарат']}\n"
            analysis += f"   Назначений: {row['Частота назначений']}\n"

            if 'Средняя цена' in row and pd.notna(row['Средняя цена']):
                analysis += f"   Средняя цена: {row['Средняя цена']:.2f} руб.\n"
            if 'Типичная дозировка' in row and pd.notna(row['Типичная дозировка']):
                analysis += f"   Дозировка: {row['Типичная дозировка']}\n"
            analysis += "\n"

        return analysis.strip()

    def _most_common_treatments(self) -> str:
        treatments = self.processor.get_most_common_treatments(top_n=20)

        if treatments is None or len(treatments) == 0:
            return "Данные о назначениях отсутствуют"

        analysis = "Наиболее частые схемы лечения\n\n"

        for idx, row in treatments.iterrows():
            analysis += f"{idx + 1}. {row['Диагноз']}\n"
            analysis += f"   Препарат: {row['Препарат']}\n"
            analysis += f"   Частота: {row['Частота']} назначений\n"
            if 'Средняя цена' in row and pd.notna(row['Средняя цена']):
                analysis += f"   Средняя цена: {row['Средняя цена']:.2f} руб.\n"
            analysis += "\n"

        return analysis.strip()

    def _seasonal_analysis(self) -> str:
        seasonal_data = self.processor.analyze_seasonality()

        if seasonal_data is None:
            return "Данные о сезонности недоступны"

        analysis = "Анализ сезонности заболеваемости\n\n"

        analysis += f"Проанализировано {seasonal_data['total_prescriptions']:,} рецептов\n\n"

        monthly = seasonal_data['monthly_stats']
        max_month = monthly.loc[monthly['количество_рецептов'].idxmax()]
        min_month = monthly.loc[monthly['количество_рецептов'].idxmin()]

        analysis += f"Пик заболеваемости: {max_month['месяц_название']} ({max_month['количество_рецептов']:,} рецептов)\n"
        analysis += f"Минимум: {min_month['месяц_название']} ({min_month['количество_рецептов']:,} рецептов)\n\n"

        analysis += "Рекомендации:\n"
        analysis += "- Подготовка ресурсов к осенне-зимнему периоду\n"
        analysis += "- Профилактические программы перед пиком заболеваемости\n"
        analysis += "- Плановые операции в летний период\n"

        return analysis.strip()

    def _select_tool(self, query: str) -> str:
        query_lower = query.lower()

        if any(word in query_lower for word in ['чем лечить', 'лечение', 'препарат для', 'лекарство']):
            keywords = ['грипп', 'орви', 'диабет', 'гипертония', 'астма', 'бронхит', 'пневмония', 'ангина', 'аллергия']
            for keyword in keywords:
                if keyword in query_lower:
                    return self._drug_recommendations(keyword)
            return self._analyze_from_merged(query_lower)

        elif any(word in query_lower for word in ['частые схемы', 'частые назначения', 'популярные препараты']):
            return self._most_common_treatments()

        elif any(word in query_lower for word in ['женщин', 'мужчин', 'пол', 'гендер']):
            return self._gender_disease_comparison()

        elif any(word in query_lower for word in ['сезонн', 'тренд', 'осень', 'зима', 'весна', 'лето']):
            return self._seasonal_analysis()

        else:
            return self._analyze_from_merged(query_lower)

    def _generate_response(self, tool_response: str, query: str) -> str:
        prompt = f"""Контекст:
                    {self.data_context}
                    
                    Данные анализа:
                    {tool_response}
                    
                    Вопрос: {query}
                    
                    Проанализируй данные и дай четкий структурированный ответ с конкретными цифрами."""

        try:
            result = GigaChatConfig.generate_text(self.pipeline, prompt, max_length=2048)
            return result.strip()

        except Exception as e:
            print(f"API error: {e}")
            return tool_response

    def query(self, user_question: str) -> Dict[str, Any]:
        print(f"\nProcessing: {user_question}")

        try:
            tool_result = self._select_tool(user_question)
            answer = self._generate_response(tool_result, user_question)

            return {
                "status": "success",
                "answer": answer,
                "model": "google/gemini-2.0-flash-exp",
                "steps": 1
            }

        except Exception as e:
            return {
                "status": "error",
                "answer": f"Ошибка: {str(e)}",
                "model": "google/gemini-2.0-flash-exp",
                "error": str(e)
            }
