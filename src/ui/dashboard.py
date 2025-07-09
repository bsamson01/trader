import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import requests
import json
from typing import Dict, Any, List

def create_dashboard():
    """Create and configure the Dash dashboard."""
    
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("📊 Strategy Analyzer Tool", className="text-center mb-4"),
                html.Hr()
            ])
        ]),
        
        # File Upload Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📁 Upload Data"),
                    dbc.CardBody([
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select a CSV File')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            multiple=False
                        ),
                        html.Div(id='upload-status')
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Analysis Controls
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("⚙️ Analysis Controls"),
                    dbc.CardBody([
                        dbc.Button("Run Analysis", id="run-analysis", color="primary", className="me-2"),
                        dbc.Button("Clear Data", id="clear-data", color="secondary", className="me-2"),
                        html.Div(id="analysis-status")
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Data Info
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📊 Data Information"),
                    dbc.CardBody(id="data-info")
                ])
            ])
        ], className="mb-4"),
        
        # Strategy Performance
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📈 Strategy Performance"),
                    dbc.CardBody([
                        dcc.Graph(id="performance-chart")
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Detailed Results
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🔍 Detailed Results"),
                    dbc.CardBody(id="detailed-results")
                ])
            ])
        ], className="mb-4"),
        
        # Trade Analysis
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📋 Trade Analysis"),
                    dbc.CardBody([
                        dcc.Graph(id="trade-analysis-chart")
                    ])
                ])
            ])
        ], className="mb-4"),
        
        # Trades History
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📊 Trades History"),
                    dbc.CardBody([
                        html.Div(id="trades-history-table")
                    ])
                ])
            ])
        ], className="mb-4")
        
    ], fluid=True)
    
    # Callbacks
    @app.callback(
        Output('upload-status', 'children'),
        Input('upload-data', 'contents'),
        State('upload-data', 'filename')
    )
    def upload_file(contents, filename):
        if contents is None:
            return ""
        
        try:
            # Decode the uploaded file
            import base64
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            # Save to temporary file and upload via API
            with open('/tmp/upload.csv', 'wb') as f:
                f.write(decoded)
            
            # Upload via API
            with open('/tmp/upload.csv', 'rb') as f:
                files = {'file': (filename, f, 'text/csv')}
                response = requests.post('http://localhost:8000/api/upload', files=files)
            
            if response.status_code == 200:
                data = response.json()
                return dbc.Alert(
                    f"✅ File uploaded successfully! {data['rows']} rows loaded.",
                    color="success"
                )
            else:
                return dbc.Alert(
                    f"❌ Upload failed: {response.text}",
                    color="danger"
                )
                
        except Exception as e:
            return dbc.Alert(f"❌ Error: {str(e)}", color="danger")
    
    @app.callback(
        Output('analysis-status', 'children'),
        Input('run-analysis', 'n_clicks')
    )
    def run_analysis(n_clicks):
        if n_clicks is None:
            return ""
        
        try:
            response = requests.post('http://localhost:8000/api/analyze')
            if response.status_code == 200:
                data = response.json()
                return dbc.Alert(
                    f"✅ Analysis completed! {data['strategies_analyzed']} strategies analyzed.",
                    color="success"
                )
            else:
                return dbc.Alert(
                    f"❌ Analysis failed: {response.text}",
                    color="danger"
                )
        except Exception as e:
            return dbc.Alert(f"❌ Error: {str(e)}", color="danger")
    
    @app.callback(
        Output('data-info', 'children'),
        Input('upload-data', 'contents'),
        Input('run-analysis', 'n_clicks')
    )
    def update_data_info(contents, n_clicks):
        try:
            response = requests.get('http://localhost:8000/api/data/info')
            if response.status_code == 200:
                data = response.json()
                return [
                    html.P(f"📅 Date Range: {data['date_range']['start']} to {data['date_range']['end']}"),
                    html.P(f"📊 Total Rows: {data['total_rows']:,}"),
                    html.P(f"⏱️ Timeframe: {data['timeframe']}"),
                    html.P(f"💱 Symbol: {data['symbol']}")
                ]
            else:
                return html.P("No data uploaded")
        except:
            return html.P("No data uploaded")
    
    @app.callback(
        Output('performance-chart', 'figure'),
        Input('run-analysis', 'n_clicks')
    )
    def update_performance_chart(n_clicks):
        if n_clicks is None:
            return go.Figure()
        
        try:
            response = requests.get('http://localhost:8000/api/results')
            if response.status_code == 200:
                data = response.json()
                
                # Extract performance data
                strategies = []
                profits = []
                win_rates = []
                
                for strategy_name, result in data['individual_results'].items():
                    if 'metrics' in result and result['metrics']:
                        strategies.append(strategy_name.replace('_', ' ').title())
                        profits.append(result['metrics'].get('total_profit', 0))
                        win_rates.append(result['metrics'].get('win_rate', 0) * 100)
                
                # Create bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=strategies,
                    y=profits,
                    name='Total Profit',
                    marker_color='lightblue'
                ))
                
                fig.update_layout(
                    title='Strategy Performance Comparison',
                    xaxis_title='Strategy',
                    yaxis_title='Total Profit',
                    height=400
                )
                
                return fig
            else:
                return go.Figure()
        except:
            return go.Figure()
    
    @app.callback(
        Output('detailed-results', 'children'),
        Input('run-analysis', 'n_clicks')
    )
    def update_detailed_results(n_clicks):
        if n_clicks is None:
            return ""
        
        try:
            response = requests.get('http://localhost:8000/api/results')
            if response.status_code == 200:
                data = response.json()
                
                results_cards = []
                for strategy_name, result in data['individual_results'].items():
                    if 'metrics' in result and result['metrics']:
                        metrics = result['metrics']
                        card = dbc.Card([
                            dbc.CardHeader(strategy_name.replace('_', ' ').title()),
                            dbc.CardBody([
                                html.P(f"Total Trades: {metrics.get('total_trades', 0)}"),
                                html.P(f"Win Rate: {metrics.get('win_rate', 0):.1%}"),
                                html.P(f"Total Profit: {metrics.get('total_profit', 0):.6f}"),
                                html.P(f"Avg R-Multiple: {metrics.get('avg_r_multiple', 0):.2f}"),
                                html.P(f"Max Drawdown: {metrics.get('max_drawdown', 0):.6f}")
                            ])
                        ], className="mb-2")
                        results_cards.append(card)
                
                return results_cards
            else:
                return html.P("No results available")
        except:
            return html.P("No results available")
    
    @app.callback(
        Output('trade-analysis-chart', 'figure'),
        Input('run-analysis', 'n_clicks')
    )
    def update_trade_analysis(n_clicks):
        if n_clicks is None:
            return go.Figure()
        
        try:
            response = requests.get('http://localhost:8000/api/results')
            if response.status_code == 200:
                data = response.json()
                
                # Collect all trades
                all_trades = []
                for strategy_name, result in data['individual_results'].items():
                    if 'trades' in result and result['trades']:
                        for trade in result['trades']:
                            trade['strategy'] = strategy_name
                            all_trades.append(trade)
                
                if all_trades:
                    df_trades = pd.DataFrame(all_trades)
                    
                    # Create scatter plot of profit vs duration
                    fig = px.scatter(
                        df_trades,
                        x='duration',
                        y='profit',
                        color='strategy',
                        title='Trade Analysis: Profit vs Duration',
                        labels={'duration': 'Duration (minutes)', 'profit': 'Profit'}
                    )
                    
                    fig.update_layout(height=400)
                    return fig
                else:
                    return go.Figure()
            else:
                return go.Figure()
        except:
            return go.Figure()
    
    @app.callback(
        Output('clear-data', 'n_clicks'),
        Input('clear-data', 'n_clicks')
    )
    def clear_data(n_clicks):
        if n_clicks:
            try:
                requests.delete('http://localhost:8000/api/clear')
            except:
                pass
        return None
    
    @app.callback(
        Output('trades-history-table', 'children'),
        Input('run-analysis', 'n_clicks')
    )
    def update_trades_history(n_clicks):
        if n_clicks is None:
            return ""
        
        try:
            response = requests.get('http://localhost:8000/api/trades')
            if response.status_code == 200:
                data = response.json()
                
                if not data.get('trades'):
                    return html.P("No trades found")
                
                # Create table header
                table_header = [
                    html.Thead(html.Tr([
                        html.Th("Strategy"),
                        html.Th("Entry Time"),
                        html.Th("Entry Price"),
                        html.Th("Exit Time"),
                        html.Th("Exit Price"),
                        html.Th("Side"),
                        html.Th("Profit/Loss"),
                        html.Th("Duration")
                    ]))
                ]
                
                # Create table body
                table_rows = []
                for trade in data['trades']:
                    # Determine color based on profit/loss
                    profit_loss = trade.get('profit', 0)
                    color = 'green' if profit_loss > 0 else 'red' if profit_loss < 0 else 'black'
                    
                    # Format duration
                    duration_minutes = trade.get('duration', 0)
                    duration_str = f"{duration_minutes:.0f} min"
                    
                    # Format side
                    side = "LONG" if trade.get('side') == 1 else "SHORT"
                    
                    row = html.Tr([
                        html.Td(trade.get('strategy', '').replace('_', ' ').title()),
                        html.Td(trade.get('entry_time', '')),
                        html.Td(f"{trade.get('entry_price', 0):.5f}"),
                        html.Td(trade.get('exit_time', '')),
                        html.Td(f"{trade.get('exit_price', 0):.5f}"),
                        html.Td(side),
                        html.Td(
                            f"{profit_loss:.6f}",
                            style={'color': color, 'fontWeight': 'bold'}
                        ),
                        html.Td(duration_str)
                    ])
                    table_rows.append(row)
                
                table_body = html.Tbody(table_rows)
                
                # Create the table
                table = dbc.Table(
                    [table_header[0], table_body],
                    bordered=True,
                    hover=True,
                    responsive=True,
                    striped=True,
                    className="mt-3"
                )
                
                return [
                    html.H5(f"Total Trades: {len(data['trades'])}"),
                    html.Hr(),
                    table
                ]
            else:
                return html.P("No trades data available")
        except Exception as e:
            return html.P(f"Error loading trades: {str(e)}")
    
    return app