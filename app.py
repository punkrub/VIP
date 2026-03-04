import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
from pycaret.classification import load_model, predict_model

model = load_model('vehicle_model')
df = pd.read_csv('cleaned_accident.csv')

provinces = sorted(df['Province'].unique())
sexes = df['Sex'].unique()

app = dash.Dash(__name__)

app.layout = html.Div(style={'fontFamily': 'Tahoma', 'padding': '20px', 'backgroundColor': '#f4f6f9'}, children=[
    html.H2("V-Predict", style={'textAlign': 'center', 'color': '#2c3e50'}),
    html.Div(style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '2px 2px 10px #ddd'}, children=[
        html.Label("อายุ (ปี):"),
        dcc.Input(id='input-age', type='number', value=30, style={'margin': '10px', 'padding': '5px'}),
        html.Label("เพศ:"),
        dcc.Dropdown(id='input-sex', options=[{'label': s, 'value': s} for s in sexes], value=sexes[0], style={'margin': '10px'}),
        html.Label("จังหวัดที่เกิดเหตุ:"),
        dcc.Dropdown(id='input-province', options=[{'label': p, 'value': p} for p in provinces], value=provinces[0], style={'margin': '10px'}),
        html.Button('พยากรณ์', id='btn-predict', style={'marginTop': '15px', 'padding': '10px 20px', 'backgroundColor': '#3498db', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
    ]),
    html.Div(id='output-prediction', style={'textAlign': 'center', 'marginTop': '20px'}),
    html.Div(style={'marginTop': '30px', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px'}, children=[
        html.H3("ข้อมูลเชิงลึกรายจังหวัด"),
        html.Div(id='insight-text', style={'padding': '15px', 'backgroundColor': '#e8f4f8', 'borderRadius': '8px', 'marginBottom': '10px', 'fontWeight': 'bold', 'color': '#2980b9'}),
        html.Button("📥 โหลดข้อมูลสถิติจังหวัดนี้", id="btn-download", style={'marginBottom': '10px', 'padding': '8px 15px', 'backgroundColor': '#27ae60', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
        dcc.Download(id="download-dataframe-csv"),
        dcc.Graph(id='graph-extra')
    ])
])

@app.callback(
    [Output('output-prediction', 'children'), Output('graph-extra', 'figure'), Output('insight-text', 'children')],
    [Input('btn-predict', 'n_clicks'), Input('input-province', 'value')],
    [State('input-age', 'value'), State('input-sex', 'value')]
)
def update_dash(n_clicks, province, age, sex):
    prov_data = df[df['Province'] == province]
    fig = px.pie(prov_data, names='Vehicle', title=f'สัดส่วนการเกิดอุบัติเหตุแยกตามพาหนะ จังหวัด {province}', hole=0.3)
    
    top_vehicle = prov_data['Vehicle'].value_counts().idxmax()
    max_count = prov_data['Vehicle'].value_counts().max()
    total = len(prov_data)
    percent = (max_count / total) * 100 if total > 0 else 0
    insight = f"Insight: จากข้อมูล {total} เคสในจังหวัด{province} พบว่า '{top_vehicle}' เกิดอุบัติเหตุสูงสุดถึง {percent:.1f}%"
    
    ctx = dash.callback_context
    if not ctx.triggered or ctx.triggered[0]['prop_id'] == 'input-province.value' or age is None:
        return html.H3("กรุณาระบุข้อมูลและกดปุ่มพยากรณ์", style={'color': '#e74c3c'}), fig, insight
        
    input_df = pd.DataFrame({'Age': [age], 'Sex': [sex], 'Province': [province]})
    pred = predict_model(model, data=input_df)
    
    result = pred['prediction_label'].iloc[0]
    score = pred['prediction_score'].iloc[0] * 100
    color = "green" if score > 70 else "orange" if score > 50 else "red"
    
    prediction_text = html.Div([
        html.H3(f"พยากรณ์ยานพาหนะ: {result}", style={'color': '#2c3e50', 'margin': '5px'}),
        html.Span(f"(ความมั่นใจของ AI: {score:.2f}%)", style={'color': color, 'fontSize': '18px', 'fontWeight': 'bold'})
    ])
    
    return prediction_text, fig, insight

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn-download", "n_clicks"),
    State('input-province', 'value'),
    prevent_initial_call=True
)
def download_csv(n_clicks, province):
    prov_data = df[df['Province'] == province]
    return dcc.send_data_frame(prov_data.to_csv, f"accident_stat_{province}.csv", index=False)

if __name__ == '__main__':
    app.run(debug=True)