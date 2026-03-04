import pandas as pd
from pycaret.classification import setup, compare_models, finalize_model, save_model

df = pd.read_csv('_2568 (1).csv')

df_clean = df[['Age', 'Sex', 'จ.ที่เสียชีวิต', 'Vehicle Merge Final']].dropna()
df_clean.rename(columns={'จ.ที่เสียชีวิต': 'Province', 'Vehicle Merge Final': 'Vehicle'}, inplace=True)
df_clean = df_clean[df_clean['Vehicle'] != 'ไม่ระบุพาหนะ']

top_vehicles = ['รถจักรยานยนต์', 'รถยนต์', 'คนเดินเท้า', 'รถจักรยาน']
df_clean = df_clean[df_clean['Vehicle'].isin(top_vehicles)]

df_clean.to_csv('cleaned_accident.csv', index=False)

clf_setup = setup(data=df_clean, target='Vehicle', session_id=42, verbose=False)
best_model = compare_models()
final_model = finalize_model(best_model)

save_model(final_model, 'vehicle_model')