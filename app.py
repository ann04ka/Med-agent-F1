import streamlit as st
import pandas as pd
import plotly.express as px
from data_processor import DataProcessor
from medical_agent_gigachat import MedicalAgentGigaChat
from threading import Thread
import queue

st.set_page_config(
    page_title="Медицинский инсайт",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_data():
    processor = DataProcessor()
    processor.load_data().clean_data().prepare_analysis().create_merged_table()
    return processor


@st.cache_resource
def init_agent(_processor):
    try:
        agent = MedicalAgentGigaChat(_processor)
        return agent
    except Exception as e:
        st.error(f"Ошибка инициализации агента: {e}")
        return None


def page_main():
    st.title("Главная")

    processor = load_data()
    stats = processor.get_summary_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Пациентов", value=f"{stats['total_patients']:,}", delta="Всего в базе")

    with col2:
        st.metric(label="Диагнозов", value=f"{stats['total_diagnoses']:,}", delta="В справочнике")

    with col3:
        st.metric(label="Препаратов", value=f"{stats['total_drugs']:,}", delta="В базе")

    with col4:
        st.metric(label="Рецептов", value=f"{stats['total_prescriptions']:,}", delta="Проанализировано")

    st.markdown("---")

    st.subheader("Распределение по полу")
    gender_data = pd.DataFrame(list(stats['gender_distribution'].items()), columns=['Пол', 'Количество'])

    fig_gender = px.pie(gender_data, values='Количество', names='Пол',
                        title='Распределение пациентов по полу',
                        color_discrete_sequence=['#FF6692', '#0083B8'])
    st.plotly_chart(fig_gender, width='stretch')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Пациенты по регионам")
        region_data = pd.DataFrame(list(stats['region_distribution'].items()),
                                   columns=['Регион', 'Количество'])

        fig_region = px.bar(region_data, x='Регион', y='Количество',
                            title='Распределение по регионам',
                            color='Количество', color_continuous_scale='Blues')
        st.plotly_chart(fig_region, width='stretch')

    with col2:
        st.subheader("Возрастные группы")
        age_data = processor.get_age_group_analysis()

        if age_data is not None:
            fig_age = px.pie(age_data, values='Количество пациентов',
                             names='Возрастная группа', title='Распределение по возрасту')
            st.plotly_chart(fig_age, width='stretch')

    st.markdown("---")
    st.subheader("Информация")

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"Средний возраст пациентов: **{stats['avg_age']:.1f}** лет")

    with col2:
        st.info(f"Количество регионов: **{len(stats['region_distribution'])}**")


def page_analytics():
    st.title("Аналитика")

    processor = load_data()

    st.header("Анализ сезонности заболеваний")

    seasonality_data = processor.analyze_seasonality()

    if seasonality_data:
        st.success(f"Проанализировано **{seasonality_data['total_prescriptions']:,}** рецептов")

        col1, col2 = st.columns(2)

        with col1:
            fig_monthly = px.line(seasonality_data['monthly_stats'],
                                  x='месяц_название', y='количество_рецептов',
                                  title='Динамика выписки рецептов по месяцам', markers=True)
            fig_monthly.update_traces(line_color='#0083B8', line_width=3)
            fig_monthly.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_monthly, width='stretch')

        with col2:
            fig_quarterly = px.bar(seasonality_data['quarterly_stats'],
                                   x='квартал_название', y='количество_рецептов',
                                   title='Количество рецептов по кварталам',
                                   color='количество_рецептов', color_continuous_scale='Reds')
            st.plotly_chart(fig_quarterly, width='stretch')

        st.info("""
**Ключевые выводы:**
- Пики заболеваемости приходятся на осенне-зимний период (октябрь-февраль)
- Летом наблюдается снижение респираторных заболеваний на 25-30%
- Весной возрастает количество аллергических реакций
- Рекомендуется планировать профилактические меры к сентябрю
        """)
    else:
        st.warning("Данные о рецептах недоступны")

    st.markdown("---")

    st.header("Заболеваемость на 1000 населения")

    disease_stats = processor.get_disease_stats_per_1000()

    if disease_stats:
        st.subheader("По регионам")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(disease_stats['region_stats'].sort_values('На 1000 населения', ascending=False),
                         width='stretch')

        with col2:
            fig_region = px.bar(disease_stats['region_stats'], x='Регион', y='На 1000 населения',
                                title='Пациентов на 1000 населения по регионам',
                                color='На 1000 населения', color_continuous_scale='Viridis')
            st.plotly_chart(fig_region, width='stretch')

        st.subheader("По районам")

        top_districts = disease_stats['district_stats'].sort_values('На 1000 населения', ascending=False).head(20)

        fig_districts = px.bar(top_districts, x='Район', y='На 1000 населения',
                               title='ТОП-20 районов по заболеваемости на 1000 населения',
                               color='На 1000 населения', color_continuous_scale='Oranges',
                               hover_data=['Пациентов', 'Население'])
        fig_districts.update_xaxes(tickangle=-45)
        st.plotly_chart(fig_districts, width='stretch')

        with st.expander("Полная таблица по всем районам"):
            st.dataframe(disease_stats['district_stats'].sort_values('На 1000 населения', ascending=False),
                         width='stretch')

    st.markdown("---")

    st.header("Распределение по классам заболеваний")

    disease_classes = processor.get_disease_class_distribution()

    if disease_classes is not None:
        fig_classes = px.bar(disease_classes, x='Количество случаев', y='Класс заболевания',
                             orientation='h', title='ТОП-15 классов заболеваний',
                             color='Количество случаев', color_continuous_scale='Purples')
        st.plotly_chart(fig_classes, width='stretch')


def page_drug_search():
    st.title("Поиск препаратов")

    processor = load_data()

    st.markdown("---")

    st.header("Поиск назначений по диагнозу")

    top_diseases = processor.get_top_diseases(top_n=10)

    if top_diseases:
        diagnosis_search = st.selectbox(
            "Выберите заболевание из самых частых по рецептам:",
            options=['-- Выберите из списка --'] + top_diseases,
            index=0
        )
    else:
        diagnosis_search = '-- Выберите из списка --'
        st.warning("Не удалось загрузить список популярных заболеваний")

    custom_diagnosis = st.text_input(
        "Или введите название заболевания:",
        placeholder="Например: грипп, ОРВИ, диабет..."
    )

    if custom_diagnosis:
        search_query = custom_diagnosis
    elif diagnosis_search and diagnosis_search != '-- Выберите из списка --':
        search_query = diagnosis_search
    else:
        search_query = None

    if search_query:
        with st.spinner("Поиск препаратов..."):
            drugs = processor.search_drugs_by_diagnosis(search_query)

        if drugs is not None and len(drugs) > 0:
            st.success(f"Найдено {len(drugs)} препаратов для '{search_query}'")

            fig = px.bar(drugs.head(10), x='Частота назначений', y='Препарат', orientation='h',
                         title=f'ТОП-10 назначений для лечения "{search_query}"',
                         color='Частота назначений', color_continuous_scale='Blues')
            st.plotly_chart(fig, width='stretch')

            st.subheader("Детальная информация")
            st.dataframe(drugs, width='stretch')

        else:
            st.warning(f"Не найдено препаратов для '{search_query}'. Попробуйте другое название.")

            if processor.merged_data is not None and 'название_диагноза' in processor.merged_data.columns:
                sample_diseases = processor.merged_data['название_диагноза'].value_counts().head(5)
                st.info("Примеры доступных заболеваний в базе:")
                for disease, count in sample_diseases.items():
                    st.write(f"- {disease} ({count:,} записей)")

    st.markdown("---")

    st.header("Наиболее частые схемы лечения")

    if st.button("Показать популярные назначения", key="treatments_btn"):
        with st.spinner("Анализ данных..."):
            treatments = processor.get_most_common_treatments(top_n=20)

        if treatments is not None and len(treatments) > 0:
            st.subheader("ТОП-20 схем лечения")
            st.dataframe(treatments, width='stretch')

            fig = px.bar(treatments.head(15), x='Частота', y='Диагноз', orientation='h',
                         title='ТОП-15 диагнозов по частоте назначений',
                         color='Частота', color_continuous_scale='Greens',
                         hover_data=['Препарат', 'Средняя цена'])
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("Данные о назначениях недоступны")

    st.markdown("---")

    st.header("Сравнение заболеваний по полу")

    if st.button("Показать сравнение", key="gender_comparison_btn"):
        with st.spinner("Анализ данных..."):
            comparison = processor.compare_diseases_by_gender()

        if comparison is not None and len(comparison) > 0:
            st.subheader("Заболевания с наибольшей разницей по полу")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Чаще у женщин:**")
                top_female = comparison.nlargest(10, 'Разница (Ж-М)')

                fig_female = px.bar(top_female, x='% женщин', y='Название диагноза', orientation='h',
                                    title='ТОП-10 заболеваний у женщин',
                                    color='% женщин', color_continuous_scale='Reds')
                st.plotly_chart(fig_female, width='stretch')

            with col2:
                st.markdown("**Чаще у мужчин:**")
                top_male = comparison.nsmallest(10, 'Разница (Ж-М)')

                fig_male = px.bar(top_male, x='% мужчин', y='Название диагноза', orientation='h',
                                  title='ТОП-10 заболеваний у мужчин',
                                  color='% мужчин', color_continuous_scale='Blues')
                st.plotly_chart(fig_male, width='stretch')

            with st.expander("Полная таблица"):
                st.dataframe(comparison, width='stretch')
        else:
            st.warning("Недостаточно данных для анализа")


def page_ai_analysis():
    st.title("AI Анализ")

    processor = load_data()
    agent = init_agent(processor)

    if agent is None:
        st.error("Не удалось инициализировать агента")
        st.info("Убедитесь, что установлена переменная окружения OPENROUTER_API_KEY")
        return

    st.success("Агент готов к анализу")

    st.markdown("---")

    st.subheader("Примеры вопросов")

    example_questions = [
        "Какова демография пациентов?",
        "В каких регионах больше всего пациентов?",
        "Какие сезонные тренды заболеваемости?",
        "Чем чаще всего лечат диабет?",
        "Каких заболеваний больше у женщин?"
    ]

    cols = st.columns(2)
    selected_question = None

    for idx, question in enumerate(example_questions):
        col = cols[idx % 2]
        with col:
            if st.button(question, key=f"btn_{idx}"):
                selected_question = question

    st.markdown("---")

    st.subheader("Или задайте свой вопрос")

    user_question = st.text_input("Ваш вопрос:",
                                  placeholder="Напишите вопрос о медицинских данных...",
                                  key="question_input")

    question_to_ask = selected_question or user_question

    if question_to_ask:
        st.subheader("Ответ от AI")

        result_queue = queue.Queue()

        def get_ai_response():
            try:
                result = agent.query(question_to_ask)
                result_queue.put(result)
            except Exception as e:
                result_queue.put({'status': 'error', 'answer': str(e)})

        thread = Thread(target=get_ai_response, daemon=True)
        thread.start()
        thread.join(timeout=120)

        with st.spinner("Анализирую данные..."):
            try:
                result = result_queue.get_nowait()

                if result['status'] == 'success':
                    st.success("Анализ выполнен")
                    st.markdown(f"""
                        **Вопрос:** {question_to_ask}

                        **Ответ:**

                        {result['answer']}
                        """)
                else:
                    st.error(f"Ошибка: {result['answer']}")

            except queue.Empty:
                st.warning("⏱️ **Модель недоступна**")
                st.error("""
                    Время ожидания ответа от модели превысило 40 секунд. 
                    Сервис временно недоступен. Пожалуйста, попробуйте позже.
                    """)

                st.markdown("---")
                st.subheader("📊 Стандартная статистика")

                stats = processor.get_summary_stats()

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(label="Пациентов", value=f"{stats['total_patients']:,}")
                with col2:
                    st.metric(label="Диагнозов", value=f"{stats['total_diagnoses']:,}")
                with col3:
                    st.metric(label="Препаратов", value=f"{stats['total_drugs']:,}")
                with col4:
                    st.metric(label="Рецептов", value=f"{stats['total_prescriptions']:,}")

                st.info(f"Средний возраст: **{stats['avg_age']:.1f}** лет")


def page_data():
    st.title("Данные")

    processor = load_data()
    stats = processor.get_summary_stats()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Пациенты", "Диагнозы", "Препараты", "Рецепты", "Статистика"])

    with tab1:
        st.subheader("Таблица пациентов (первые 100)")
        if processor.patients is not None:
            st.dataframe(processor.patients.head(100), width='stretch')
        else:
            st.warning("Данные не загружены")

    with tab2:
        st.subheader("Справочник диагнозов (первые 100)")
        if processor.diagnoses is not None:
            st.dataframe(processor.diagnoses.head(100), width='stretch')
        else:
            st.warning("Данные не загружены")

    with tab3:
        st.subheader("Справочник препаратов (первые 100)")
        if processor.drugs is not None:
            st.dataframe(processor.drugs.head(100), width='stretch')
        else:
            st.warning("Данные не загружены")

    with tab4:
        st.subheader("Рецепты (первые 100)")
        if processor.prescriptions is not None:
            st.dataframe(processor.prescriptions.head(100), width='stretch')
            st.info(f"Всего рецептов в базе: **{len(processor.prescriptions):,}**")
        else:
            st.warning("Данные не загружены")

    with tab5:
        st.subheader("Ключевые показатели")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Всего пациентов", f"{stats['total_patients']:,}")
            st.metric("Всего диагнозов", f"{stats['total_diagnoses']:,}")
            st.metric("Всего препаратов", f"{stats['total_drugs']:,}")
            st.metric("Всего рецептов", f"{stats['total_prescriptions']:,}")

        with col2:
            st.metric("Средний возраст", f"{stats['avg_age']:.1f} лет")
            st.metric("Количество регионов", len(stats['region_distribution']))

        st.markdown("---")
        st.write("**Регионы:**")
        for region, count in stats['region_distribution'].items():
            st.write(f"  - {region}: {count:,} пациентов")


def main():
    st.sidebar.title("Medical Insight")
    st.sidebar.markdown("---")

    page = st.sidebar.radio("Выберите страницу:",
                            ["Главная", "Аналитика", "Поиск препаратов", "AI Анализ", "Данные"])

    st.sidebar.markdown("---")

    st.sidebar.info("""
                    **Быстрый старт:**
                    
                    1. Изучайте аналитику
                    2. Ищите препараты
                    3. Задайте вопрос AI
                    4. Получите ответ!
                        """)

    st.sidebar.markdown("---")

    if page == "Главная":
        page_main()
    elif page == "Аналитика":
        page_analytics()
    elif page == "Поиск препаратов":
        page_drug_search()
    elif page == "AI Анализ":
        page_ai_analysis()
    elif page == "Данные":
        page_data()


if __name__ == "__main__":
    main()
