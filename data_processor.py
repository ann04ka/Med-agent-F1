import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

SPB_POPULATION = {
    'АДМИРАЛТЕЙСКИЙ': 154424,
    'ВАСИЛЕОСТРОВСКИЙ': 208720,
    'ВЫБОРГСКИЙ': 551925,
    'КАЛИНИНСКИЙ': 532972,
    'КИРОВСКИЙ': 342658,
    'КОЛПИНСКИЙ': 146822,
    'КРАСНОГВАРДЕЙСКИЙ': 366435,
    'КРАСНОСЕЛЬСКИЙ': 443752,
    'КРОНШТАДТСКИЙ': 44500,
    'КУРОРТНЫЙ': 83700,
    'МОСКОВСКИЙ': 383759,
    'НЕВСКИЙ': 548800,
    'ПЕТРОГРАДСКИЙ': 143537,
    'ПЕТРОДВОРЦОВЫЙ': 135449,
    'ПРИМОРСКИЙ': 704000,
    'ПУШКИНСКИЙ': 234366,
    'ФРУНЗЕНСКИЙ': 424682,
    'ЦЕНТРАЛЬНЫЙ': 226674
}

LO_POPULATION = {
    'Бокситогорский': 51751,
    'Волосовский': 50376,
    'Волховский': 80768,
    'Всеволожский': 519360,
    'Выборгский': 196905,
    'Гатчинский': 263942,
    'Кингисеппский': 84937,
    'Киришский': 60865,
    'Кировский': 109506,
    'Лодейнопольский': 27851,
    'Ломоносовский': 79079,
    'Лужский': 76969,
    'Подпорожский': 26147,
    'Приозерский': 57597,
    'Сланцевский': 45902,
    'Сосновоборский': 65367,
    'Тихвинский': 67475,
    'Тосненский': 136200
}

REGION_POPULATION = {
    'Санкт-Петербург': 5598473,
    'Ленинградская область': 2035762
}

ALL_POPULATION = {**SPB_POPULATION, **LO_POPULATION}


class DataProcessor:

    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.patients = None
        self.diagnoses = None
        self.drugs = None
        self.prescriptions = None
        self.merged_data = None
        self.population_data = ALL_POPULATION
        self.region_population = REGION_POPULATION

    def load_data(self):
        print("Loading data...")

        self.patients = pd.read_csv(self.data_dir / 'данные_пациентов.csv')
        self.diagnoses = pd.read_csv(self.data_dir / 'данные_диагнозы.csv')
        self.drugs = pd.read_csv(self.data_dir / 'данные_препараты.csv')
        self.prescriptions = pd.read_csv(self.data_dir / 'данные_рецептов.csv')

        print(f"Loaded: {len(self.patients):,} patients, {len(self.diagnoses):,} diagnoses, "
              f"{len(self.drugs):,} drugs, {len(self.prescriptions):,} prescriptions")

        return self

    def clean_data(self):
        print("Cleaning data...")

        self.patients = self.patients.drop_duplicates()
        self.diagnoses = self.diagnoses.drop_duplicates()
        self.drugs = self.drugs.drop_duplicates()
        self.prescriptions = self.prescriptions.drop_duplicates()

        print("Data cleaned")
        return self

    def prepare_analysis(self):
        print("Preparing analysis...")

        self.patients['дата_рождения'] = pd.to_datetime(self.patients['дата_рождения'], errors='coerce')
        self.patients['возраст'] = (pd.Timestamp.now() - self.patients['дата_рождения']).dt.days // 365

        def age_group(age):
            if pd.isna(age):
                return 'Неизвестно'
            elif age < 18:
                return '0-17 (Дети)'
            elif age < 30:
                return '18-29 (Молодые)'
            elif age < 45:
                return '30-44 (Взрослые)'
            elif age < 60:
                return '45-59 (Средний возраст)'
            else:
                return '60+ (Пожилые)'

        self.patients['возрастная_группа'] = self.patients['возраст'].apply(age_group)

        print("Analysis ready")
        return self

    def create_merged_table(self):
        print("Creating merged table from prescriptions...")

        merged = self.prescriptions.copy()

        patient_cols = ['id_пациента', 'пол', 'возраст', 'возрастная_группа', 'район_проживания', 'регион']
        merged = merged.merge(
            self.patients[patient_cols],
            left_on='id_пациента_1',
            right_on='id_пациента',
            how='left',
            suffixes=('', '_patient')
        )

        diagnosis_cols = ['код_мкб', 'название_диагноза', 'класс_заболевания']
        merged = merged.merge(
            self.diagnoses[diagnosis_cols],
            left_on='код_диагноза',
            right_on='код_мкб',
            how='left',
            suffixes=('', '_diag')
        )

        drug_cols = [c for c in self.drugs.columns if
                     c in ['код_препарата', 'Торговое название', 'дозировка', 'стоимость']]

        merged = merged.merge(
            self.drugs[drug_cols],
            on='код_препарата',
            how='left',
            suffixes=('', '_drug')
        )

        if 'Торговое название' in merged.columns:
            merged = merged.rename(columns={'Торговое название': 'название_препарата'})

        merged['дата_рецепта'] = pd.to_datetime(merged['дата_рецепта'], errors='coerce')
        merged['год'] = merged['дата_рецепта'].dt.year
        merged['месяц'] = merged['дата_рецепта'].dt.month
        merged['квартал'] = merged['дата_рецепта'].dt.quarter

        merged['месяц_название'] = merged['месяц'].map({
            1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
            5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
            9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
        })

        def get_season(month):
            if pd.isna(month):
                return None
            if month in [12, 1, 2]:
                return 'Зима'
            elif month in [3, 4, 5]:
                return 'Весна'
            elif month in [6, 7, 8]:
                return 'Лето'
            else:
                return 'Осень'

        merged['сезон'] = merged['месяц'].apply(get_season)

        self.merged_data = merged
        merged.to_csv('merged.csv')

        print(f"Merged table created: {len(merged):,} records")

        return self

    def get_merged_data_summary(self):
        if self.merged_data is None:
            return None

        summary = {
            'total_records': len(self.merged_data),
            'unique_patients': self.merged_data['id_пациента'].nunique(),
            'unique_diagnoses': self.merged_data['код_диагноза'].nunique(),
            'unique_drugs': self.merged_data['код_препарата'].nunique(),
            'records_with_price': self.merged_data[
                'стоимость'].notna().sum() if 'стоимость' in self.merged_data.columns else 0,
            'total_cost': self.merged_data['стоимость'].sum() if 'стоимость' in self.merged_data.columns else 0,
            'avg_cost': self.merged_data['стоимость'].mean() if 'стоимость' in self.merged_data.columns else 0
        }

        return summary

    def get_top_diseases(self, top_n=10):
        if self.merged_data is None or 'название_диагноза' not in self.merged_data.columns:
            return []

        top = self.merged_data['название_диагноза'].value_counts().head(top_n)
        return top.index.tolist()

    def search_drugs_by_diagnosis(self, diagnosis_name: str):
        if self.merged_data is None:
            return None

        if 'название_диагноза' not in self.merged_data.columns or 'название_препарата' not in self.merged_data.columns:
            return None

        # diagnosis_name_lower = diagnosis_name.lower().strip()

        filtered = self.merged_data[
            self.merged_data['название_диагноза'].str.lower().str.contains(diagnosis_name, case=False, na=False,
                                                                           regex=False)
        ]

        if len(filtered) == 0:
            return None

        agg_dict = {'код_препарата': 'count'}
        if 'стоимость' in filtered.columns:
            agg_dict['стоимость'] = ['mean', 'min', 'max']
        if 'дозировка' in filtered.columns:
            agg_dict['дозировка'] = lambda x: x.mode()[0] if len(x.mode()) > 0 else None

        drug_stats = filtered.groupby('название_препарата', as_index=False).agg(agg_dict)

        if isinstance(drug_stats.columns, pd.MultiIndex):
            drug_stats.columns = drug_stats.columns.droplevel(0)

        new_cols = ['Препарат', 'Частота назначений']
        if 'стоимость' in agg_dict:
            new_cols.extend(['Средняя цена', 'Мин. цена', 'Макс. цена'])
        if 'дозировка' in agg_dict:
            new_cols.append('Типичная дозировка')

        drug_stats.columns = new_cols
        drug_stats = drug_stats.sort_values('Частота назначений', ascending=False)

        return drug_stats

    def get_most_common_treatments(self, top_n=20):
        if self.merged_data is None:
            return None

        required_cols = ['название_диагноза', 'название_препарата']
        if not all(col in self.merged_data.columns for col in required_cols):
            return None

        group_cols = ['название_диагноза', 'название_препарата']

        if 'стоимость' in self.merged_data.columns:
            result = self.merged_data.groupby(group_cols, as_index=False).agg({
                'код_препарата': 'count',
                'стоимость': 'mean'
            })
            result.columns = ['Диагноз', 'Препарат', 'Частота', 'Средняя цена']
        else:
            result = self.merged_data.groupby(group_cols, as_index=False).size()
            result.columns = ['Диагноз', 'Препарат', 'Частота']
            result['Средняя цена'] = None

        result = result.sort_values('Частота', ascending=False).head(top_n)

        return result

    def compare_diseases_by_gender(self):
        if self.merged_data is None:
            return None

        if 'название_диагноза' not in self.merged_data.columns or 'пол' not in self.merged_data.columns:
            return None

        gender_disease = self.merged_data.groupby(['название_диагноза', 'пол'], as_index=False).size()
        gender_disease.columns = ['название_диагноза', 'пол', 'count']

        pivot = gender_disease.pivot(index='название_диагноза', columns='пол', values='count').fillna(0)

        male_col = [c for c in pivot.columns if c.lower() == 'м'][0] if any(
            c.lower() == 'м' for c in pivot.columns) else None
        female_col = [c for c in pivot.columns if c.lower() == 'ж'][0] if any(
            c.lower() == 'ж' for c in pivot.columns) else None

        if male_col and female_col:
            pivot['Всего'] = pivot[male_col] + pivot[female_col]
            pivot['Разница (Ж-М)'] = pivot[female_col] - pivot[male_col]
            pivot['% женщин'] = (pivot[female_col] / pivot['Всего'] * 100).round(1)
            pivot['% мужчин'] = (pivot[male_col] / pivot['Всего'] * 100).round(1)
            pivot['Абс. разница'] = pivot['Разница (Ж-М)'].abs()
            pivot = pivot.sort_values('Абс. разница', ascending=False)

            result = pivot.reset_index()
            result = result.rename(columns={'название_диагноза': 'Название диагноза'})

            return result

        return None

    def get_seasonal_disease_trends(self):
        if self.merged_data is None:
            return None

        if 'сезон' not in self.merged_data.columns or 'название_диагноза' not in self.merged_data.columns:
            return None

        seasonal = self.merged_data.groupby(['сезон', 'название_диагноза'], as_index=False).size()
        seasonal.columns = ['сезон', 'название_диагноза', 'count']

        result = {}
        for season in ['Зима', 'Весна', 'Лето', 'Осень']:
            season_data = seasonal[seasonal['сезон'] == season].sort_values('count', ascending=False).head(10)
            result[season] = season_data[['название_диагноза', 'count']].to_dict('records')

        return result

    def get_top_diseases_by_group(self, group_by='пол', top_n=10):
        if self.merged_data is None:
            return None

        if group_by not in self.merged_data.columns or 'название_диагноза' not in self.merged_data.columns:
            return None

        result = {}

        for group_value in self.merged_data[group_by].dropna().unique():
            filtered = self.merged_data[self.merged_data[group_by] == group_value]
            top_diseases = filtered['название_диагноза'].value_counts().head(top_n)
            result[group_value] = top_diseases.to_dict()

        return result

    def get_drug_recommendations(self, diagnosis_name: str, top_n=10):
        drugs = self.search_drugs_by_diagnosis(diagnosis_name)

        if drugs is not None and len(drugs) > 0:
            return drugs.head(top_n)

        return None

    def get_summary_stats(self):
        total_prescriptions = len(self.prescriptions) if self.prescriptions is not None else 0

        stats = {
            'total_patients': len(self.patients),
            'avg_age': self.patients['возраст'].mean(),
            'gender_distribution': self.patients['пол'].value_counts().to_dict(),
            'region_distribution': self.patients['регион'].value_counts().to_dict(),
            'distr_distribution': self.patients['район_проживания'].value_counts().to_dict(),
            'total_diagnoses': len(self.diagnoses),
            'total_drugs': len(self.drugs),
            'total_prescriptions': total_prescriptions
        }

        return stats

    def get_disease_stats_per_1000(self):
        if self.patients is None:
            return None

        district_stats = []
        for district in self.patients['район_проживания'].unique():
            if pd.isna(district):
                continue

            patient_count = len(self.patients[self.patients['район_проживания'] == district])
            population = self.population_data.get(district, 0)

            if population > 0:
                per_1000 = (patient_count / population) * 1000
                district_stats.append({
                    'Район': district,
                    'Пациентов': patient_count,
                    'Население': population,
                    'На 1000 населения': round(per_1000, 2)
                })

        region_stats = []
        for region in self.patients['регион'].unique():
            if pd.isna(region):
                continue

            patient_count = len(self.patients[self.patients['регион'] == region])
            population = self.region_population.get(region, 0)

            if population > 0:
                per_1000 = (patient_count / population) * 1000
                region_stats.append({
                    'Регион': region,
                    'Пациентов': patient_count,
                    'Население': population,
                    'На 1000 населения': round(per_1000, 2)
                })

        return {
            'district_stats': pd.DataFrame(district_stats),
            'region_stats': pd.DataFrame(region_stats)
        }

    def analyze_seasonality(self):
        if self.merged_data is None:
            return None

        if 'месяц' not in self.merged_data.columns:
            return None

        monthly_stats = self.merged_data.groupby('месяц', as_index=False).size()
        monthly_stats.columns = ['месяц', 'количество_рецептов']
        monthly_stats['месяц_название'] = monthly_stats['месяц'].map({
            1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
            5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
            9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
        })

        quarterly_stats = self.merged_data.groupby('квартал', as_index=False).size()
        quarterly_stats.columns = ['квартал', 'количество_рецептов']
        quarterly_stats['квартал_название'] = quarterly_stats['квартал'].map({
            1: 'Q1 (Янв-Мар)', 2: 'Q2 (Апр-Июн)',
            3: 'Q3 (Июл-Сен)', 4: 'Q4 (Окт-Дек)'
        })

        return {
            'monthly_stats': monthly_stats,
            'quarterly_stats': quarterly_stats,
            'total_prescriptions': len(self.merged_data[self.merged_data['дата_рецепта'].notna()])
        }

    def get_disease_class_distribution(self):
        if self.merged_data is None:
            return None

        if 'класс_заболевания' not in self.merged_data.columns:
            return None

        class_dist = self.merged_data['класс_заболевания'].value_counts().head(15)

        return pd.DataFrame({
            'Класс заболевания': class_dist.index,
            'Количество случаев': class_dist.values
        })

    def get_age_group_analysis(self):
        if self.patients is None:
            return None

        age_dist = self.patients['возрастная_группа'].value_counts()

        return pd.DataFrame({
            'Возрастная группа': age_dist.index,
            'Количество пациентов': age_dist.values
        })
