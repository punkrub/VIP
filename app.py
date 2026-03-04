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
    html.H2("ระบบพยากรณ์ยานพาหนะที่เกิดอุบัติเหตุ", style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    html.Div(style={'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px', 'boxShadow': '2px 2px 10px #ddd'}, children=[
        html.Label("อายุ (ปี):"),
        dcc.Input(id='input-age', type='number', value=30, style={'margin': '10px', 'padding': '5px'}),
        
        html.Label("เพศ:"),
        dcc.Dropdown(id='input-sex', options=[{'label': s, 'value': s} for s in sexes], value=sexes[0], style={'margin': '10px'}),
        
        html.Label("จังหวัดที่เกิดเหตุ:"),
        dcc.Dropdown(id='input-province', options=[{'label': p, 'value': p} for p in provinces], value=provinces[0], style={'margin': '10px'}),
        
        html.Button('พยากรณ์', id='btn-predict', style={'marginTop': '15px', 'padding': '10px 20px', 'backgroundColor': '#3498db', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
    ]),
    
    html.H3(id='output-prediction', style={'textAlign': 'center', 'color': '#e74c3c', 'marginTop': '20px'}),
    
    html.Div(style={'marginTop': '30px', 'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px'}, children=[
        html.H3("โมดูลเสริม: สัดส่วนยานพาหนะที่เกิดอุบัติเหตุในจังหวัดที่เลือก"),
        dcc.Graph(id='graph-extra')
    ])
])

@app.callback(
    [Output('output-prediction', 'children'), Output('graph-extra', 'figure')],
    [Input('btn-predict', 'n_clicks'), Input('input-province', 'value')],
    [State('input-age', 'value'), State('input-sex', 'value')]
)
def update_dash(n_clicks, province, age, sex):
    prov_data = df[df['Province'] == province]
    fig = px.pie(prov_data, names='Vehicle', title=f'สัดส่วนการเกิดอุบัติเหตุแยกตามพาหนะ จังหวัด {province}', hole=0.3)
    
    ctx = dash.callback_context
    if not ctx.triggered or ctx.triggered[0]['prop_id'] == 'input-province.value' or age is None:
        return "กรุณาระบุข้อมูลและกดปุ่มพยากรณ์", fig
        
    input_df = pd.DataFrame({'Age': [age], 'Sex': [sex], 'Province': [province]})
    pred = predict_model(model, data=input_df)
    result = pred['prediction_label'].iloc[0]
    
    return f"พยากรณ์ว่าผู้ป่วยใช้งาน: {result}", fig

if __name__ == '__main__':
    app.run(debug=True)