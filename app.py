from flask import Flask, render_template, request
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server
import matplotlib.pyplot as plt
from prophet import Prophet
import io
import base64

app = Flask(__name__)

# ── Load & clean data once at startup ──────────────────────────
df_acc_raw = pd.read_csv('data/state_accidents.csv')
df_fat_raw = pd.read_csv('data/state_fatalities.csv')

df_acc = df_acc_raw[['State', '2019 Accidents', '2020 Accidents',
                      '2021 Accidents', '2022 Accidents', '2023 Accidents']].copy()
df_acc = df_acc.melt(id_vars='State', var_name='Year', value_name='Accidents')
df_acc['Year'] = df_acc['Year'].str.extract('(\d+)').astype(int)

df_fat = df_fat_raw[['State', '2019 Killed', '2020 Killed',
                      '2021 Killed', '2022 Killed', '2023 Killed']].copy()
df_fat = df_fat.melt(id_vars='State', var_name='Year', value_name='Fatalities')
df_fat['Year'] = df_fat['Year'].str.extract('(\d+)').astype(int)

# Clean numeric columns — remove commas and convert to int
df_acc['Accidents'] = pd.to_numeric(
    df_acc['Accidents'].astype(str).str.replace(',', ''), errors='coerce'
)
df_fat['Fatalities'] = pd.to_numeric(
    df_fat['Fatalities'].astype(str).str.replace(',', ''), errors='coerce'
)

df = pd.merge(df_acc, df_fat, on=['State', 'Year'])
df = df.dropna(subset=['State'])
df = df[df['State'] != 'All India']
df['Fatality_Rate'] = (df['Fatalities'] / df['Accidents']) * 100
df = df.sort_values(['State', 'Year']).reset_index(drop=True)

# Road user & safety device data
df_ru = pd.read_csv('data/road_user_fatalities.csv')
df_ru = df_ru[~df_ru['Road-user category'].str.contains('share|Total', case=False, na=False)]
df_ru['Persons killed 2022'] = df_ru['Persons killed 2022'].str.replace(',', '').astype(int)
df_ru['Persons killed 2023'] = df_ru['Persons killed 2023'].str.replace(',', '').astype(int)

df_sd = pd.read_csv('data/safety_devices.csv')
df_sd_cat = df_sd[df_sd['Drivers'] == 'Category'].copy()
no_helmet_killed = int(str(df_sd_cat['No Helmet - Killed'].values[0]).replace(',', ''))
no_seatbelt_killed = int(str(df_sd_cat['No Seat belt - killed'].values[0]).replace(',', ''))

ALL_STATES = sorted(df['State'].unique().tolist())

# ── Helper: matplotlib fig → base64 string ─────────────────────
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

# ── Helper: Prophet forecast for a state ───────────────────────
def get_forecast(state_name):
    state_data = df[df['State'] == state_name][['Year', 'Accidents']].copy()
    state_data.columns = ['ds', 'y']
    state_data['ds'] = pd.to_datetime(state_data['ds'].astype(str))

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95
    )
    model.fit(state_data)
    future = model.make_future_dataframe(periods=4, freq='A')
    forecast = model.predict(future)
    return state_data, forecast

# ── Routes ──────────────────────────────────────────────────────

@app.route('/')
def index():
    # National trend — Plotly
    national = df.groupby('Year')[['Accidents', 'Fatalities']].sum().reset_index()
    fig_national = go.Figure()
    fig_national.add_trace(go.Scatter(
        x=national['Year'], y=national['Accidents'],
        name='Accidents', line=dict(color='steelblue', width=3),
        mode='lines+markers', marker=dict(size=8)
    ))
    fig_national.add_trace(go.Scatter(
        x=national['Year'], y=national['Fatalities'],
        name='Fatalities', line=dict(color='crimson', width=3),
        mode='lines+markers', marker=dict(size=8),
        yaxis='y2'
    ))
    fig_national.update_layout(
        title='India Road Accidents & Fatalities — National Trend 2019–2023',
        yaxis=dict(title='Total Accidents', title_font=dict(color='steelblue')),
        yaxis2=dict(title='Total Fatalities', title_font=dict(color='crimson'),
                    overlaying='y', side='right'),
        legend=dict(x=0.01, y=0.99),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400
    )
    national_chart = pio.to_html(fig_national, full_html=False)

    # Top 10 states — Plotly
    top10 = (df.groupby('State')['Accidents']
               .mean()
               .sort_values(ascending=False)
               .head(10)
               .reset_index())
    fig_top10 = px.bar(top10, x='Accidents', y='State',
                       orientation='h',
                       color='Accidents',
                       color_continuous_scale='Reds',
                       title='Top 10 States by Average Annual Accidents (2019–2023)')
    fig_top10.update_layout(
        yaxis=dict(autorange='reversed'),
        coloraxis_showscale=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400
    )
    top10_chart = pio.to_html(fig_top10, full_html=False)

    # Headline stats
    total_2023 = int(df[df['Year'] == 2023]['Accidents'].sum())
    total_fatalities_2023 = int(df[df['Year'] == 2023]['Fatalities'].sum())
    two_wheeler_pct = 44.8

    return render_template('index.html',
                           national_chart=national_chart,
                           top10_chart=top10_chart,
                           total_2023=f"{total_2023:,}",
                           total_fatalities_2023=f"{total_fatalities_2023:,}",
                           no_helmet_killed=f"{no_helmet_killed:,}",
                           no_seatbelt_killed=f"{no_seatbelt_killed:,}",
                           two_wheeler_pct=two_wheeler_pct,
                           states=ALL_STATES)


@app.route('/state', methods=['GET', 'POST'])
def state():
    selected = request.form.get('state', ALL_STATES[0])
    state_df = df[df['State'] == selected].copy()

    # Trend line — Plotly
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=state_df['Year'], y=state_df['Accidents'],
        name='Accidents', line=dict(color='steelblue', width=3),
        mode='lines+markers', marker=dict(size=8)
    ))
    fig_trend.add_trace(go.Scatter(
        x=state_df['Year'], y=state_df['Fatalities'],
        name='Fatalities', line=dict(color='crimson', width=3),
        mode='lines+markers', marker=dict(size=8),
        yaxis='y2'
    ))
    fig_trend.update_layout(
        title=f'{selected} — Accidents & Fatalities (2019–2023)',
        yaxis=dict(title='Accidents', title_font=dict(color='steelblue')),
        yaxis2=dict(title='Fatalities', title_font=dict(color='crimson'),
                    overlaying='y', side='right'),
        plot_bgcolor='white', paper_bgcolor='white', height=380
    )
    trend_chart = pio.to_html(fig_trend, full_html=False)

    # Fatality rate — Plotly
    fig_rate = px.bar(state_df, x='Year', y='Fatality_Rate',
                      color='Fatality_Rate',
                      color_continuous_scale='YlOrRd',
                      title=f'{selected} — Fatality Rate (Deaths per 100 Accidents)')
    fig_rate.update_layout(
        coloraxis_showscale=False,
        plot_bgcolor='white', paper_bgcolor='white', height=350
    )
    rate_chart = pio.to_html(fig_rate, full_html=False)

    # Prophet forecast — Matplotlib
    actual, forecast = get_forecast(selected)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(actual['ds'], actual['y'],
            'o-', color='steelblue', label='Actual', linewidth=2)
    ax.plot(forecast['ds'], forecast['yhat'],
            '--', color='crimson', label='Forecast', linewidth=2)
    ax.fill_between(forecast['ds'],
                    forecast['yhat_lower'],
                    forecast['yhat_upper'],
                    alpha=0.2, color='crimson', label='95% Confidence Interval')
    ax.set_title(f'{selected} — Road Accident Forecast (2024–2026)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Accidents')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    forecast_img = fig_to_base64(fig)

    # Key stats
    latest = state_df[state_df['Year'] == 2023].iloc[0]
    oldest = state_df[state_df['Year'] == 2019].iloc[0]
    pct_change = ((latest['Accidents'] - oldest['Accidents']) / oldest['Accidents']) * 100
    forecast_2026 = forecast[forecast['ds'].dt.year == 2026]['yhat'].values[0]

    stats = {
        'accidents_2023': f"{int(latest['Accidents']):,}",
        'fatalities_2023': f"{int(latest['Fatalities']):,}",
        'fatality_rate_2023': f"{latest['Fatality_Rate']:.1f}%",
        'pct_change': f"{pct_change:+.1f}%",
        'forecast_2026': f"{int(forecast_2026):,}"
    }

    return render_template('state.html',
                           states=ALL_STATES,
                           selected=selected,
                           trend_chart=trend_chart,
                           rate_chart=rate_chart,
                           forecast_img=forecast_img,
                           stats=stats)


@app.route('/compare', methods=['GET', 'POST'])
def compare():
    selected_states = request.form.getlist('states')
    if not selected_states:
        selected_states = ALL_STATES[:3]

    fig_compare = go.Figure()
    for state_name in selected_states:
        state_df = df[df['State'] == state_name]
        fig_compare.add_trace(go.Scatter(
            x=state_df['Year'], y=state_df['Accidents'],
            name=state_name, mode='lines+markers',
            line=dict(width=2), marker=dict(size=7)
        ))
    fig_compare.update_layout(
        title='State-wise Accident Trend Comparison (2019–2023)',
        xaxis=dict(tickvals=[2019, 2020, 2021, 2022, 2023]),
        yaxis=dict(title='Accidents'),
        plot_bgcolor='white', paper_bgcolor='white', height=420
    )
    compare_chart = pio.to_html(fig_compare, full_html=False)

    # Forecast comparison table
    forecast_data = []
    for state_name in selected_states:
        _, forecast = get_forecast(state_name)
        val_2026 = forecast[forecast['ds'].dt.year == 2026]['yhat'].values[0]
        actual_2023 = int(df[(df['State'] == state_name) & (df['Year'] == 2023)]['Accidents'].values[0])
        forecast_data.append({
            'state': state_name,
            'actual_2023': f"{actual_2023:,}",
            'forecast_2026': f"{int(val_2026):,}",
            'change': f"{int(val_2026) - actual_2023:+,}"
        })

    return render_template('compare.html',
                           states=ALL_STATES,
                           selected_states=selected_states,
                           compare_chart=compare_chart,
                           forecast_data=forecast_data)


if __name__ == '__main__':
    app.run(debug=True)