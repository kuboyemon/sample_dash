#ライブラリのインポート
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

#線形重回帰による数値予測モデリング

#元データの読み込み(seaborn標準のtipsデータ)
df=sns.load_dataset('tips')

#データの整形（必要データのみ抽出）
use_data=df[['total_bill','size','time','tip']]
use_data=pd.get_dummies(use_data,drop_first=True)

#機械学習sklearn(重回帰)への適用
X=use_data[['total_bill','size','time_Dinner']]
Y=use_data[['tip']]
clf=LinearRegression()
clf.fit(X,Y)

#表示するグラフの作成
import plotly.graph_objects as go
from plotly._subplots import make_subplots

tip_plots=make_subplots(rows=1,cols=3,start_cell='bottom-left')
tip_plots.add_trace(go.Box(x=df['time'],y=df['tip'],name='time vs tip'),row=1, col=1)
tip_plots.add_trace(go.Scatter(x=df['total_bill'],y=df['tip'],mode='markers',name='total_bill vs tip'),row=1,col=2)
tip_plots.add_trace(go.Scatter(x=df['size'],y=df['tip'],mode='markers',name='size vs tip'),row=1,col=3)

tip_plots.update_layout(
    xaxis_title_text='Time(Lunch or Dinner)',
    yaxis_title_text='Tip ($)',
)
tip_plots.update_layout(
    xaxis2_title_text='Total Bill($)',
    yaxis2_title_text='Tip ($)',
)
tip_plots.update_layout(
    xaxis3_title_text='Size(人)',
    yaxis3_title_text='Tip ($)',
)

#アプリ部分の作成
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from dash import dash_table

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout=html.Div([
    html.H1('チップの額を予測するアプリです。',style={'textAlign':'center'}),
    html.H2('まずは分析に使うデータを見てみましょう。'),
    dash_table.DataTable(
        style_cell={'textAlign':'center','width':'150px'},
        fill_width=False,
        fixed_rows={'headers':True},
        page_size=10,
        filter_action='native',
        sort_action='native',
        columns=[{'name':col,'id':col} for col in df.columns],
        data=df.to_dict('records')
    ),
    html.P('モデリングに必要なデータは{}件ですよ'.format(len(df))),
    html.H2('次はグラフを見てみましょう（グラフの要素は固定)'),
    dcc.Graph(
        id='graph',
        figure=tip_plots,
        #CSSによるスタイル記載可能
        style={}
    ),
    html.H2('最後に予測用のデータをインプットしてみましょう'),
    dcc.Input(
        id='total_bill',
        placeholder='total bill ここに値を入れてください。',
        type='text',
        style={'width':'20%'},
        value=''
    ),
    dcc.Input(
        id='size',
        placeholder='size ここに値を入れてください。',
        type='text',
        style={'width':'20%'},
        value=''
    ),
    dcc.RadioItems(
        id='time',
        options=[
            {'label':'ランチ','value':'Lunch'},
            {'label':'ディナー','value':'Dinner'}
        ],
        value='Lunch',
        labelStyle={'display':'inline-block'}
    ),
    html.Button(
        id='submit_button',
        n_clicks=0,
        children='Submit'
    ),
    html.H2('チップの予測額はいくらかな？'),
    html.Div(
        id='output_pred',
        style={'tectAlign':'center','fontSize':30,'color':'red'}
    )
])

#コールバック部分の作成
@app.callback(
    Output('output_pred','children'),
    Input('submit_button','n_clicks'),
    [State('total_bill','value'),
    State('size','value'),
    State('time','value')]
)
def prediction(n_clicks,total_bill,size,time):
    if time=='Lunch':
        dinner01=0
    else:
        dinner01=1
    if (total_bill and size):
        value_df=pd.DataFrame([],columns=['Total bill','Size','Dinner flag'])
        record=pd.Series([total_bill,size,dinner01],index=value_df.columns,dtype='float64')
        value_df=value_df.append(record,ignore_index=True)
        Y_pred=clf.predict(value_df)
        return_text='チップ額はおそらく'+str('{:.2g}'.format(Y_pred[0,0])+'ドルくらいでしょう')
        return return_text
    else:
        return 'ちゃんと入力してね'

#アプリの実行
if __name__=='__main__':
    app.run_server(debug=True)
