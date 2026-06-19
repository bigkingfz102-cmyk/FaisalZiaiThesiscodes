from __future__ import annotations
import re
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_FILENAME = 'Bachelor thesis7_June 2, 2026_00.39.csv'
OUTPUT_DIR = SCRIPT_DIR / 'analysis_outputs'
EXPLANATION_TYPES = ['A', 'B', 'C', 'D', 'E']
EXPLANATION_TYPE_NAMES = {'A': 'Rule-based decision path', 'B': 'Local feature-attribution / key-factor explanation', 'C': 'Counterfactual explanation', 'D': 'LLM natural-language explanation', 'E': 'Doctor-written-style explanation'}
SHORT_TYPE_LABELS = {'A': 'A: Rule-based', 'B': 'B: Key-factor', 'C': 'C: Counterfactual', 'D': 'D: LLM natural-language', 'E': 'E: Doctor-written-style'}
TOPIC_NAMES = {'H': 'Headache', 'B': 'Belly/stomach pain', 'C': 'Cold/cough/sore throat', 'F': 'Fever/flu-like symptoms', 'N': 'Nausea/vomiting/diarrhea', 'S': 'Skin/rash/itching'}
CONSTRUCT_ORDER = ['trust', 'satisfaction', 'usability_understandability', 'perceived_effectiveness']
CONSTRUCT_LABELS = {'trust': 'Trust', 'satisfaction': 'Satisfaction', 'usability_understandability': 'Usability / understandability', 'perceived_effectiveness': 'Perceived effectiveness'}
TYPE_COLORS = {'A': '#4C72B0', 'B': '#55A868', 'C': '#C44E52', 'D': '#8172B2', 'E': '#CCB974'}
AGE_GROUP_ORDER = ['18-25', '25-35', '35-45', '45-50', '50-60', '60+']
GENDER_ORDER = ['Female', 'Woman', 'Male', 'Man', 'Non-binary / other', 'Other', 'Prefer not to say', 'Prefer not to answer']
MISSING_DEMOGRAPHIC_VALUES = {'', 'nan', 'none', 'na', 'n/a'}
SCENARIO_PATTERN = re.compile('\\b([HBCFNS]\\d+)\\b', re.IGNORECASE)
EXPLANATION_CODE_PATTERN = re.compile('Explanation\\s*[\\(\\[]?\\s*([HBCFNS]\\d[A-E])\\s*[\\)\\]]?', re.IGNORECASE)
EXPLANATION_CODE_FALLBACK = re.compile('\\b([HBCFNS]\\d[A-E])\\b', re.IGNORECASE)
SCENARIO_FROM_TEXT_PATTERN = re.compile('scenario\\s+([HBCFNS]\\d+)\\b', re.IGNORECASE)

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_text(value) -> str:
    if pd.isna(value):
        return ''
    text = str(value).strip().lower()
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    return text

def find_column(columns, keywords, level=0):
    for col in columns:
        if isinstance(col, tuple):
            name = str(col[level])
        else:
            name = str(col)
        name_lower = name.lower()
        if all((keyword.lower() in name_lower for keyword in keywords)):
            return col
    return None

def find_columns(columns, keywords, level=0):
    matches = []
    for col in columns:
        if isinstance(col, tuple):
            name = str(col[level])
        else:
            name = str(col)
        name_lower = name.lower()
        if all((keyword.lower() in name_lower for keyword in keywords)):
            matches.append(col)
    return matches

def find_demographic_column(columns, column_map, name_keywords, text_keywords, exclude=()):
    name_keywords = [kw.lower() for kw in name_keywords]
    text_keywords = [kw.lower() for kw in text_keywords]
    exclude = [kw.lower() for kw in exclude]
    for col in columns:
        name = str(col).strip().lower()
        if name in name_keywords:
            return col
    for col in columns:
        name = str(col).strip().lower()
        question = column_map.get(col, '').lower()
        if any((bad in name or bad in question for bad in exclude)):
            continue
        if any((keyword in question for keyword in text_keywords)):
            return col
    return None

def column_text(col, header_df) -> str:
    if isinstance(col, tuple):
        parts = [str(part).strip() for part in col if str(part).strip() not in ('', 'nan')]
        return ' | '.join(parts)
    return str(col)

def parse_explanation_code(text: str):
    if not text:
        return None
    match = EXPLANATION_CODE_PATTERN.search(text)
    if match:
        return match.group(1).upper()
    match = EXPLANATION_CODE_FALLBACK.search(text)
    if match:
        return match.group(1).upper()
    return None

def parse_scenario_from_text(text: str):
    if not text:
        return None
    match = SCENARIO_FROM_TEXT_PATTERN.search(text)
    if match:
        return match.group(1).upper()
    return None

def scenario_to_topic(scenario: str) -> str:
    if not scenario:
        return ''
    topic_code = scenario[0].upper()
    return TOPIC_NAMES.get(topic_code, topic_code)

def explanation_type_from_code(code: str) -> str:
    if not code:
        return ''
    return code[-1].upper()

def parse_construct(text: str):
    text_norm = normalize_text(text)
    if 'did not have this one' in text_norm:
        return 'validation'
    if 'trust this explanation' in text_norm:
        return 'trust'
    if 'satisfied with this explanation' in text_norm:
        return 'satisfaction'
    if 'easy to use and understand' in text_norm:
        return 'usability_understandability'
    if 'understand what i should do next' in text_norm:
        return 'perceived_effectiveness'
    return None

def to_numeric_rating(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text == '':
        return np.nan
    match = re.match('^(\\d+)', text)
    if match:
        rating = float(match.group(1))
        if 1 <= rating <= 7:
            return rating
    try:
        rating = float(text)
        if 1 <= rating <= 7:
            return rating
    except ValueError:
        return np.nan
    return np.nan

def is_truthy(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return value == 1
    text = str(value).strip().lower()
    return text in {'1', 'true', 'yes', 'y'}

def is_preview_row(row, status_col) -> bool:
    if status_col is None:
        return False
    status_text = normalize_text(row[status_col])
    return 'preview' in status_text

def is_finished_row(row, finished_col) -> bool:
    if finished_col is None:
        return True
    return is_truthy(row[finished_col])

def is_consent_row(row, consent_col) -> bool:
    if consent_col is None:
        return True
    consent_text = normalize_text(row[consent_col])
    positive_words = ('agree', 'consent', 'yes', 'accept', 'i consent')
    negative_words = ('disagree', 'no', 'decline', 'do not consent')
    if any((word in consent_text for word in negative_words)):
        return False
    if consent_text == '':
        return False
    return any((word in consent_text for word in positive_words)) or is_truthy(row[consent_col])

def compute_mean_table(df, group_cols, value_col='rating'):
    summary = df.groupby(group_cols, dropna=False)[value_col].agg(['mean', 'std', 'count']).reset_index().rename(columns={'mean': 'mean_rating', 'std': 'sd', 'count': 'n'})
    summary['sem'] = summary['sd'] / np.sqrt(summary['n'].replace(0, np.nan))
    summary['ci95_low'] = summary['mean_rating'] - 1.96 * summary['sem']
    summary['ci95_high'] = summary['mean_rating'] + 1.96 * summary['sem']
    return summary

def ordered_pair(type_a: str, type_b: str):
    return tuple(sorted([type_a, type_b]))

def pair_label(type_a: str, type_b: str) -> str:
    first, second = ordered_pair(type_a, type_b)
    return f'{first}|{second}'

def holm_correction(p_values):
    p_values = np.asarray(p_values, dtype=float)
    n_tests = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n_tests, dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        adjusted_value = min(1.0, (n_tests - rank) * p_values[index])
        running_max = max(running_max, adjusted_value)
        adjusted[index] = running_max
    return adjusted

def load_qualtrics_csv(csv_path: Path):
    print(f'Loading CSV: {csv_path}')
    raw_df = pd.read_csv(csv_path, header=[0, 1, 2], low_memory=False)
    flat_names = []
    seen = {}
    column_map = {}
    for col in raw_df.columns:
        internal_name = str(col[0]).strip() if isinstance(col, tuple) else str(col).strip()
        full_text = column_text(col, raw_df)
        if internal_name in seen:
            seen[internal_name] += 1
            unique_name = f'{internal_name}__{seen[internal_name]}'
        else:
            seen[internal_name] = 0
            unique_name = internal_name
        flat_names.append(unique_name)
        column_map[unique_name] = full_text
    raw_df.columns = flat_names
    return (raw_df, column_map)

def clean_participants(raw_df, column_map):
    df = raw_df.copy()
    total_raw_rows = len(df)
    status_col = find_column(df.columns, ['status'])
    finished_col = find_column(df.columns, ['finished'])
    consent_col = find_column(df.columns, ['consent'])
    if consent_col is None:
        for col in df.columns:
            if 'consent' in column_map.get(col, '').lower():
                consent_col = col
                break
    response_col = find_column(df.columns, ['responseid'])
    age_col = find_demographic_column(df.columns, column_map, name_keywords=['age'], text_keywords=['what is your age', 'age:', 'age category', 'your age'], exclude=['language', 'consent', 'percentage', 'average'])
    gender_col = find_demographic_column(df.columns, column_map, name_keywords=['gender', 'sex'], text_keywords=['gender', 'sex/gender'], exclude=['language', 'consent'])
    preview_removed = 0
    if status_col is not None:
        preview_mask = df.apply(lambda row: is_preview_row(row, status_col), axis=1)
        preview_removed = int(preview_mask.sum())
        df = df.loc[~preview_mask].copy()
    finished_rows = len(df)
    if finished_col is not None:
        df = df.loc[df.apply(lambda row: is_finished_row(row, finished_col), axis=1)].copy()
    consent_rows = len(df)
    if consent_col is not None:
        df = df.loc[df.apply(lambda row: is_consent_row(row, consent_col), axis=1)].copy()
    if response_col is not None:
        df['response_id'] = df[response_col].astype(str)
    else:
        df['response_id'] = [f'row_{i}' for i in range(len(df))]
    unique_ids = df['response_id'].drop_duplicates().tolist()
    id_map = {response_id: f'P{index:03d}' for index, response_id in enumerate(unique_ids, start=1)}
    df['participant_id'] = df['response_id'].map(id_map)
    if age_col is not None:
        df['age'] = df[age_col].astype(str)
    else:
        df['age'] = np.nan
    if gender_col is not None:
        df['gender'] = df[gender_col].astype(str)
    else:
        df['gender'] = np.nan
    cleaning_info = {'total_raw_rows': total_raw_rows, 'preview_rows_removed': preview_removed, 'finished_rows': finished_rows, 'consent_rows': consent_rows, 'cleaned_participant_rows': len(df), 'response_col': response_col, 'age_col': age_col, 'gender_col': gender_col}
    return (df, cleaning_info)

def extract_codes_from_value(value):
    if pd.isna(value):
        return []
    text = str(value)
    return [match.upper() for match in EXPLANATION_CODE_FALLBACK.findall(text)]

def build_selected_explanations(df, column_map):
    scenario_columns = {}
    for col in df.columns:
        col_text = column_map.get(col, '')
        internal_name = col[0].lower() if isinstance(col, tuple) else str(col).lower()
        if 'choose2explanations' not in internal_name and 'which two explanations' not in col_text.lower():
            continue
        scenario = parse_scenario_from_text(col_text)
        if scenario is None:
            continue
        scenario_columns.setdefault(scenario, []).append(col)
    selected_rows = []
    for _, row in df.iterrows():
        participant_id = row['participant_id']
        response_id = row['response_id']
        for scenario, cols in scenario_columns.items():
            selected_codes = []
            for col in cols:
                value = row[col]
                col_text = column_map.get(col, '')
                if is_truthy(value):
                    selected_codes.extend(extract_codes_from_value(col_text))
                elif not pd.isna(value) and str(value).strip() != '':
                    selected_codes.extend(extract_codes_from_value(value))
                    selected_codes.extend(extract_codes_from_value(col_text))
            scenario_prefix = scenario.upper()
            selected_codes = [code for code in dict.fromkeys(selected_codes) if code.startswith(scenario_prefix)]
            if len(selected_codes) == 0:
                continue
            selected_rows.append({'participant_id': participant_id, 'response_id': response_id, 'scenario': scenario, 'topic': scenario_to_topic(scenario), 'selected_codes': selected_codes})
    if not selected_rows:
        return pd.DataFrame(columns=['participant_id', 'response_id', 'scenario', 'topic', 'selected_codes'])
    return pd.DataFrame(selected_rows)

def selected_lookup(selected_df):
    lookup = {}
    for _, row in selected_df.iterrows():
        lookup[row['participant_id'], row['scenario']] = row['selected_codes']
    return lookup

def build_ratings_long(df, column_map, selected_df):
    selected_codes_lookup = selected_lookup(selected_df)
    rating_rows = []
    for col in df.columns:
        col_text = column_map.get(col, '')
        col_text_lower = col_text.lower()
        internal_name = col[0] if isinstance(col, tuple) else str(col)
        is_rating_column = 'rate explanation' in col_text_lower or re.search('_[A-E]_Ratings', str(internal_name), re.IGNORECASE)
        if not is_rating_column:
            continue
        explanation_code = parse_explanation_code(col_text)
        if explanation_code is None:
            explanation_code = parse_explanation_code(str(internal_name))
        if explanation_code is None:
            continue
        scenario = explanation_code[:-1].upper()
        explanation_type = explanation_type_from_code(explanation_code)
        construct_source = col_text
        if isinstance(col, tuple) and len(col) > 1:
            construct_source = str(col[1])
        construct = parse_construct(construct_source)
        if construct is None or construct == 'validation':
            continue
        for _, row in df.iterrows():
            participant_id = row['participant_id']
            selected_codes = selected_codes_lookup.get((participant_id, scenario), [])
            if explanation_code not in selected_codes:
                continue
            rating = to_numeric_rating(row[col])
            if pd.isna(rating):
                continue
            rating_rows.append({'participant_id': participant_id, 'response_id': row['response_id'], 'age': row.get('age', np.nan), 'gender': row.get('gender', np.nan), 'scenario': scenario, 'topic': scenario_to_topic(scenario), 'explanation_code': explanation_code, 'explanation_type': explanation_type, 'explanation_type_name': EXPLANATION_TYPE_NAMES[explanation_type], 'construct': construct, 'rating': rating})
    ratings_df = pd.DataFrame(rating_rows)
    return ratings_df

def is_no_preference_value(value) -> bool:
    text = normalize_text(value)
    return 'no clear preference' in text or 'no preference' in text or text in {'no', 'none', 'geen voorkeur'}

def build_preferences_long(df, column_map, selected_df):
    selected_codes_lookup = selected_lookup(selected_df)
    preference_cols = []
    for col in df.columns:
        col_text = column_map.get(col, '').lower()
        if 'prefer' in col_text and 'scenario' in col_text:
            preference_cols.append(col)
    preference_rows = []
    skipped_inconsistent = 0
    for col in preference_cols:
        col_text = column_map.get(col, '')
        scenario = parse_scenario_from_text(col_text)
        if scenario is None:
            continue
        for _, row in df.iterrows():
            participant_id = row['participant_id']
            response_id = row['response_id']
            shown_codes = selected_codes_lookup.get((participant_id, scenario), [])
            if len(shown_codes) != 2:
                skipped_inconsistent += 1
                continue
            shown_codes = sorted(shown_codes)
            shown_type_1 = explanation_type_from_code(shown_codes[0])
            shown_type_2 = explanation_type_from_code(shown_codes[1])
            raw_value = row[col]
            if pd.isna(raw_value) or str(raw_value).strip() == '':
                skipped_inconsistent += 1
                continue
            no_clear_preference = is_no_preference_value(raw_value)
            preferred_code = None
            preferred_type = None
            if not no_clear_preference:
                extracted_codes = extract_codes_from_value(raw_value)
                scenario_codes = [code for code in extracted_codes if code.startswith(scenario)]
                if len(scenario_codes) == 1:
                    preferred_code = scenario_codes[0]
                elif len(extracted_codes) == 1:
                    preferred_code = extracted_codes[0]
                else:
                    skipped_inconsistent += 1
                    continue
                if preferred_code not in shown_codes:
                    skipped_inconsistent += 1
                    continue
                preferred_type = explanation_type_from_code(preferred_code)
            preference_rows.append({'participant_id': participant_id, 'response_id': response_id, 'scenario': scenario, 'topic': scenario_to_topic(scenario), 'shown_code_1': shown_codes[0], 'shown_code_2': shown_codes[1], 'shown_type_1': shown_type_1, 'shown_type_2': shown_type_2, 'pair': pair_label(shown_type_1, shown_type_2), 'preferred_code': preferred_code, 'preferred_type': preferred_type, 'preferred_type_name': EXPLANATION_TYPE_NAMES.get(preferred_type, '') if preferred_type else '', 'no_clear_preference': no_clear_preference})
    preferences_df = pd.DataFrame(preference_rows)
    preferences_df.attrs['skipped_inconsistent'] = skipped_inconsistent
    return preferences_df

def add_type_names(summary_df):
    if 'explanation_type' in summary_df.columns:
        summary_df['explanation_type_name'] = summary_df['explanation_type'].map(EXPLANATION_TYPE_NAMES)
    return summary_df

def build_summary_tables(ratings_df, preferences_df):
    tables = {}
    tables['overall_means_by_explanation_type.csv'] = add_type_names(compute_mean_table(ratings_df, ['explanation_type']))
    tables['means_by_type_and_construct.csv'] = add_type_names(compute_mean_table(ratings_df, ['explanation_type', 'construct']))
    tables['means_by_topic_and_type.csv'] = add_type_names(compute_mean_table(ratings_df, ['topic', 'explanation_type']))
    tables['means_by_scenario_and_type.csv'] = add_type_names(compute_mean_table(ratings_df, ['scenario', 'topic', 'explanation_type']))
    shown_records = []
    for _, row in preferences_df.iterrows():
        shown_records.append({'explanation_type': row['shown_type_1']})
        shown_records.append({'explanation_type': row['shown_type_2']})
    shown_df = pd.DataFrame(shown_records)
    preferred_df = preferences_df.loc[~preferences_df['no_clear_preference'] & preferences_df['preferred_type'].notna(), ['preferred_type']].rename(columns={'preferred_type': 'explanation_type'})
    shown_counts = shown_df['explanation_type'].value_counts().rename('times_shown')
    preferred_counts = preferred_df['explanation_type'].value_counts().rename('times_preferred')
    preference_counts = pd.DataFrame({'explanation_type': EXPLANATION_TYPES}).merge(shown_counts, on='explanation_type', how='left').merge(preferred_counts, on='explanation_type', how='left')
    preference_counts['times_shown'] = preference_counts['times_shown'].fillna(0).astype(int)
    preference_counts['times_preferred'] = preference_counts['times_preferred'].fillna(0).astype(int)
    preference_counts['preference_percentage'] = np.where(preference_counts['times_shown'] > 0, preference_counts['times_preferred'] / preference_counts['times_shown'] * 100, np.nan)
    preference_counts = add_type_names(preference_counts)
    tables['preference_counts_by_type.csv'] = preference_counts
    pairwise_rows = []
    for i, type_1 in enumerate(EXPLANATION_TYPES):
        for type_2 in EXPLANATION_TYPES[i + 1:]:
            subset = preferences_df.loc[preferences_df['pair'] == pair_label(type_1, type_2)].copy()
            total_valid = len(subset)
            no_clear = int(subset['no_clear_preference'].sum())
            decided = subset.loc[~subset['no_clear_preference']]
            type_1_wins = int((decided['preferred_type'] == type_1).sum())
            type_2_wins = int((decided['preferred_type'] == type_2).sum())
            pairwise_rows.append({'type_1': type_1, 'type_2': type_2, 'type_1_wins': type_1_wins, 'type_2_wins': type_2_wins, 'no_clear_preference': no_clear, 'total_valid_preferences': total_valid, 'type_1_win_percentage': type_1_wins / total_valid * 100 if total_valid > 0 else np.nan, 'type_2_win_percentage': type_2_wins / total_valid * 100 if total_valid > 0 else np.nan})
    pairwise_df = pd.DataFrame(pairwise_rows)
    tables['pairwise_preference_counts.csv'] = pairwise_df
    matrix = pd.DataFrame(index=EXPLANATION_TYPES, columns=EXPLANATION_TYPES, dtype=float)
    for type_1 in EXPLANATION_TYPES:
        for type_2 in EXPLANATION_TYPES:
            if type_1 == type_2:
                matrix.loc[type_1, type_2] = np.nan
                continue
            pair_rows = preferences_df.loc[preferences_df['pair'] == pair_label(type_1, type_2)]
            decided = pair_rows.loc[~pair_rows['no_clear_preference']]
            if len(decided) == 0:
                matrix.loc[type_1, type_2] = np.nan
                continue
            wins = int((decided['preferred_type'] == type_1).sum())
            matrix.loc[type_1, type_2] = wins / len(decided) * 100
    tables['pairwise_preference_matrix.csv'] = matrix
    tables['participant_level_means.csv'] = add_type_names(ratings_df.groupby(['participant_id', 'explanation_type'], as_index=False).agg(mean_rating=('rating', 'mean'), n_ratings=('rating', 'count')))
    tables['participant_level_means_by_construct.csv'] = add_type_names(ratings_df.groupby(['participant_id', 'explanation_type', 'construct'], as_index=False).agg(mean_rating=('rating', 'mean'), n_ratings=('rating', 'count')))
    return tables

def build_data_quality_summary(cleaning_info, ratings_df, preferences_df):
    summary = pd.DataFrame([{'metric': 'total_raw_rows', 'value': cleaning_info['total_raw_rows']}, {'metric': 'preview_rows_removed', 'value': cleaning_info['preview_rows_removed']}, {'metric': 'finished_rows', 'value': cleaning_info['finished_rows']}, {'metric': 'consent_rows', 'value': cleaning_info['consent_rows']}, {'metric': 'cleaned_participant_rows', 'value': cleaning_info['cleaned_participant_rows']}, {'metric': 'valid_rating_rows', 'value': len(ratings_df)}, {'metric': 'valid_preference_rows', 'value': len(preferences_df)}, {'metric': 'unique_participants_in_cleaned_ratings', 'value': ratings_df['participant_id'].nunique() if len(ratings_df) else 0}, {'metric': 'unique_participants_in_cleaned_preferences', 'value': preferences_df['participant_id'].nunique() if len(preferences_df) else 0}, {'metric': 'skipped_inconsistent_preferences', 'value': preferences_df.attrs.get('skipped_inconsistent', 0)}])
    return summary

def normalize_demographic_series(series):
    cleaned = series.astype(str).str.strip()
    is_missing = cleaned.str.lower().isin(MISSING_DEMOGRAPHIC_VALUES)
    return cleaned.mask(is_missing)

def order_categories(counts, preferred_order):
    present = list(counts.index)
    ordered = [label for label in preferred_order if label in present]
    leftovers = [label for label in present if label not in ordered]
    leftovers = sorted(leftovers, key=lambda label: (-int(counts[label]), str(label)))
    return ordered + leftovers

def make_demographic_counts(df, column, group_column_name, preferred_order=None):
    values = normalize_demographic_series(df[column]).dropna()
    counts = values.value_counts()
    if preferred_order is not None and len(counts) > 0:
        counts = counts.reindex(order_categories(counts, preferred_order))
    total_non_missing = int(counts.sum())
    counts_df = pd.DataFrame({group_column_name: list(counts.index), 'count': counts.values.astype(int)})
    if total_non_missing > 0:
        counts_df['percentage'] = counts_df['count'] / total_non_missing * 100
    else:
        counts_df['percentage'] = 0.0
    return (counts_df, total_non_missing)

def compute_demographics(cleaned_df):
    participants = cleaned_df.drop_duplicates(subset='participant_id')
    total_participants = len(participants)
    age_counts, age_total = make_demographic_counts(participants, 'age', 'age_group', preferred_order=AGE_GROUP_ORDER)
    has_gender = 'gender' in participants.columns and normalize_demographic_series(participants['gender']).notna().any()
    if has_gender:
        gender_counts, gender_total = make_demographic_counts(participants, 'gender', 'gender', preferred_order=GENDER_ORDER)
    else:
        gender_counts, gender_total = (None, 0)
    return {'total_participants': total_participants, 'age_counts': age_counts, 'age_total': age_total, 'gender_counts': gender_counts, 'gender_total': gender_total, 'has_gender': has_gender}

def save_demographic_tables(demographics):
    ensure_output_dir()
    demographics['age_counts'].to_csv(OUTPUT_DIR / 'demographic_age_counts.csv', index=False)
    if demographics['has_gender']:
        demographics['gender_counts'].to_csv(OUTPUT_DIR / 'demographic_gender_counts.csv', index=False)

def _format_distribution_lines(counts_df, group_column):
    lines = []
    for _, row in counts_df.iterrows():
        lines.append(f"  {row[group_column]}: {int(row['count'])} participants ({row['percentage']:.1f}%)")
    return lines

def build_demographic_summary_text(demographics):
    lines = ['Demographic summary', '=' * 60, '']
    lines.append(f"Total valid participants used for demographic analysis: {demographics['total_participants']}")
    lines.append('')
    lines.append(f"Age distribution (based on {demographics['age_total']} answers):")
    lines.extend(_format_distribution_lines(demographics['age_counts'], 'age_group'))
    lines.append('')
    if demographics['has_gender']:
        lines.append(f"Gender distribution (based on {demographics['gender_total']} answers):")
        lines.extend(_format_distribution_lines(demographics['gender_counts'], 'gender'))
    else:
        lines.append('Gender distribution: no gender data available.')
    lines.append('')
    age_df = demographics['age_counts']
    if len(age_df) > 0:
        top_age = age_df.loc[age_df['count'].idxmax()]
        age_sentence = f"The most common age group was {top_age['age_group']} ({int(top_age['count'])} participants, {top_age['percentage']:.1f}%)."
    else:
        age_sentence = 'No age data was available.'
    gender_df = demographics['gender_counts']
    if demographics['has_gender'] and gender_df is not None and (len(gender_df) > 0):
        top_gender = gender_df.loc[gender_df['count'].idxmax()]
        gender_sentence = f" The largest gender group was {top_gender['gender']} ({int(top_gender['count'])} participants, {top_gender['percentage']:.1f}%)."
    else:
        gender_sentence = ' No gender data was available.'
    lines.append('Suggested thesis text:')
    lines.append(f"  A total of {demographics['total_participants']} valid participants were included in the analysis. " + age_sentence + gender_sentence)
    lines.append('')
    return '\n'.join(lines)

def save_demographic_summary(demographics):
    ensure_output_dir()
    text = build_demographic_summary_text(demographics)
    with open(OUTPUT_DIR / 'demographic_summary.txt', 'w', encoding='utf-8') as file:
        file.write(text)
    print('Saved demographic summary.')

def print_demographic_summary(demographics):
    print('\nDemographic summary:')
    print('Age distribution:')
    for _, row in demographics['age_counts'].iterrows():
        print(f"  {row['age_group']}: {int(row['count'])} participants ({row['percentage']:.1f}%)")
    print('Gender distribution:')
    if demographics['has_gender']:
        for _, row in demographics['gender_counts'].iterrows():
            print(f"  {row['gender']}: {int(row['count'])} participants ({row['percentage']:.1f}%)")
    else:
        print('  No gender data available.')

def setup_plot_style() -> None:
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('ggplot')
    sns.set_context('notebook', font_scale=1.1)

def save_figure(fig, filename):
    ensure_output_dir()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f'Saved figure: {path}')

def draw_demographic_bars(ax, counts_df, group_column, title, xlabel, palette_name):
    categories = counts_df[group_column].astype(str).tolist()
    counts = counts_df['count'].tolist()
    percentages = counts_df['percentage'].tolist()
    colors = sns.color_palette(palette_name, n_colors=max(len(categories), 1))
    bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=0.6)
    label_offset = max(counts) * 0.02 if counts else 0.1
    for bar, count, percentage in zip(bars, counts, percentages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + label_offset, f'{int(count)}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Number of participants')
    ax.set_ylim(0, max(counts) * 1.18 if counts else 1)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=15)
    for label in ax.get_xticklabels():
        label.set_ha('right')

def prepare_data():
    ensure_output_dir()
    csv_path = SCRIPT_DIR / CSV_FILENAME
    if not csv_path.exists():
        raise FileNotFoundError(f'Could not find the CSV file:\n  {csv_path}\nPlace the Qualtrics export in the same folder as this script.')
    raw_df, column_map = load_qualtrics_csv(csv_path)
    cleaned_df, cleaning_info = clean_participants(raw_df, column_map)
    selected_df = build_selected_explanations(cleaned_df, column_map)
    ratings_df = build_ratings_long(cleaned_df, column_map, selected_df)
    preferences_df = build_preferences_long(cleaned_df, column_map, selected_df)
    tables = build_summary_tables(ratings_df, preferences_df)
    return (cleaning_info, ratings_df, preferences_df, tables, cleaned_df)

def run_friedman_test(participant_means):
    wide = participant_means.pivot(index='participant_id', columns='explanation_type', values='mean_rating')
    complete = wide.dropna()
    if len(complete) < 3:
        return None
    try:
        result = stats.friedmanchisquare(complete['A'], complete['B'], complete['C'], complete['D'], complete['E'])
        return {'statistic': result.statistic, 'p_value': result.pvalue, 'n_participants': len(complete)}
    except Exception as exc:
        warnings.warn(f'Friedman test could not be run: {exc}')
        return None

def run_pairwise_wilcoxon(participant_means, friedman_p_value):
    wide = participant_means.pivot(index='participant_id', columns='explanation_type', values='mean_rating')
    rows = []
    raw_p_values = []
    for i, type_1 in enumerate(EXPLANATION_TYPES):
        for type_2 in EXPLANATION_TYPES[i + 1:]:
            paired = wide[[type_1, type_2]].dropna()
            if len(paired) < 5:
                rows.append({'type_1': type_1, 'type_2': type_2, 'n_pairs': len(paired), 'statistic': np.nan, 'p_value_raw': np.nan, 'p_value_holm': np.nan, 'significant_holm_0.05': False, 'note': 'Not enough paired observations'})
                raw_p_values.append(np.nan)
                continue
            try:
                result = stats.wilcoxon(paired[type_1], paired[type_2])
                rows.append({'type_1': type_1, 'type_2': type_2, 'n_pairs': len(paired), 'statistic': result.statistic, 'p_value_raw': result.pvalue, 'p_value_holm': np.nan, 'significant_holm_0.05': False, 'note': ''})
                raw_p_values.append(result.pvalue)
            except Exception as exc:
                rows.append({'type_1': type_1, 'type_2': type_2, 'n_pairs': len(paired), 'statistic': np.nan, 'p_value_raw': np.nan, 'p_value_holm': np.nan, 'significant_holm_0.05': False, 'note': str(exc)})
                raw_p_values.append(np.nan)
    wilcoxon_df = pd.DataFrame(rows)
    valid_mask = wilcoxon_df['p_value_raw'].notna()
    if valid_mask.any():
        adjusted = holm_correction(wilcoxon_df.loc[valid_mask, 'p_value_raw'].values)
        wilcoxon_df.loc[valid_mask, 'p_value_holm'] = adjusted
        wilcoxon_df.loc[valid_mask, 'significant_holm_0.05'] = adjusted < 0.05
    if friedman_p_value is None or friedman_p_value >= 0.05:
        wilcoxon_df['note'] = wilcoxon_df['note'].where(wilcoxon_df['note'] != '', 'Friedman test not significant; post-hoc results are exploratory')
    return wilcoxon_df

def run_mixed_effects_model(ratings_df):
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        warnings.warn('statsmodels is not installed; mixed-effects model skipped.')
        return None
    model_df = ratings_df.dropna(subset=['rating', 'participant_id']).copy()
    if len(model_df) < 20:
        warnings.warn('Not enough rating rows for mixed-effects model.')
        return None
    try:
        model = smf.mixedlm('rating ~ C(explanation_type) + C(construct)', model_df, groups=model_df['participant_id'])
        result = model.fit(method='lbfgs', maxiter=200, disp=False)
        return result.summary().as_text()
    except Exception as exc:
        warnings.warn(f'Mixed-effects model failed: {exc}')
        return None

def run_statistical_tests(ratings_df, tables):
    ensure_output_dir()
    participant_means = tables['participant_level_means.csv']
    friedman_result = run_friedman_test(participant_means)
    friedman_p = None if friedman_result is None else friedman_result['p_value']
    wilcoxon_df = run_pairwise_wilcoxon(participant_means, friedman_p)
    wilcoxon_df.to_csv(OUTPUT_DIR / 'pairwise_wilcoxon_tests.csv', index=False)
    mixed_model_summary = run_mixed_effects_model(ratings_df)
    lines = ['Statistical tests', '=' * 60, '']
    if friedman_result is None:
        lines.append('Friedman test: could not be run (not enough complete participants).')
    else:
        lines.extend(['Friedman test (participant-level mean ratings across A-E)', f"  Chi-square statistic: {friedman_result['statistic']:.4f}", f"  p-value: {friedman_result['p_value']:.6f}", f"  Participants included: {friedman_result['n_participants']}", ''])
    lines.append('Pairwise Wilcoxon signed-rank tests saved to pairwise_wilcoxon_tests.csv')
    lines.append('')
    if mixed_model_summary is None:
        lines.append('Mixed-effects model: not available or failed to converge.')
    else:
        lines.extend(['Mixed-effects model', '-' * 60, mixed_model_summary])
    output_text = '\n'.join(lines)
    with open(OUTPUT_DIR / 'statistical_tests.txt', 'w', encoding='utf-8') as file:
        file.write(output_text)
    print('Saved statistical test output.')

def print_final_summary(cleaning_info, ratings_df, preferences_df, tables):
    overall = tables['overall_means_by_explanation_type.csv'].set_index('explanation_type')
    preference = tables['preference_counts_by_type.csv'].set_index('explanation_type')
    print('\n' + '=' * 70)
    print('Analysis complete.')
    print(f"Raw responses: {cleaning_info['total_raw_rows']}")
    print(f'Valid rating rows: {len(ratings_df)}')
    print(f'Valid preference rows: {len(preferences_df)}')
    print(f"Unique participants in rating analysis: {(ratings_df['participant_id'].nunique() if len(ratings_df) else 0)}")
    print(f"Unique participants in preference analysis: {(preferences_df['participant_id'].nunique() if len(preferences_df) else 0)}")
    print('\nOverall mean rating by explanation type:')
    for explanation_type in EXPLANATION_TYPES:
        if explanation_type in overall.index:
            mean_value = overall.loc[explanation_type, 'mean_rating']
            print(f'  {SHORT_TYPE_LABELS[explanation_type]}: {mean_value:.2f}')
        else:
            print(f'  {SHORT_TYPE_LABELS[explanation_type]}: no data')
    print('\nPreference percentage by explanation type:')
    for explanation_type in EXPLANATION_TYPES:
        if explanation_type in preference.index:
            pct = preference.loc[explanation_type, 'preference_percentage']
            if pd.notna(pct):
                print(f'  {SHORT_TYPE_LABELS[explanation_type]}: {pct:.1f}%')
            else:
                print(f'  {SHORT_TYPE_LABELS[explanation_type]}: no data')
        else:
            print(f'  {SHORT_TYPE_LABELS[explanation_type]}: no data')
    print(f'\nOutput files saved in:\n  {OUTPUT_DIR}')
    print('=' * 70 + '\n')
