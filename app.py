from flask import Flask, render_template, request, jsonify, send_file
import sqlite3, csv, io, os, math
from datetime import date, timedelta
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'data', 'sales.db')
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024


def connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with connect() as con:
        con.execute('''CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sale_date TEXT NOT NULL,
            product TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'General',
            units INTEGER NOT NULL DEFAULT 1,
            unit_price REAL NOT NULL DEFAULT 0,
            promotion INTEGER NOT NULL DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )''')
        count = con.execute('SELECT COUNT(*) FROM sales').fetchone()[0]
        if count == 0:
            # Seed a reproducible 24-month sample dataset so the charts work immediately.
            start = pd.Timestamp.today().normalize().replace(day=1) - pd.DateOffset(months=23)
            rows = []
            for i in range(24):
                month = start + pd.DateOffset(months=i)
                for j, product in enumerate(['Everyday Essentials', 'Home & Living', 'Tech Accessories']):
                    seasonal = 1 + 0.18 * math.sin((month.month / 12) * 2 * math.pi)
                    trend = 1 + i * 0.018
                    units = max(8, int((28 + j * 9) * seasonal * trend + ((i * 7 + j * 3) % 9)))
                    price = [12.5, 24.0, 38.0][j]
                    promo = 1 if month.month in (6, 11, 12) else 0
                    rows.append((month.strftime('%Y-%m-%d'), product, ['Daily Use', 'Home', 'Electronics'][j], units, price, promo))
            con.executemany('INSERT INTO sales (sale_date,product,category,units,unit_price,promotion) VALUES (?,?,?,?,?,?)', rows)


def all_sales():
    with connect() as con:
        rows = con.execute('SELECT * FROM sales ORDER BY sale_date DESC, id DESC').fetchall()
    return [dict(r) for r in rows]


def sales_df():
    rows = all_sales()
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=['id','sale_date','product','category','units','unit_price','promotion','revenue'])
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
    df['revenue'] = pd.to_numeric(df['units'], errors='coerce').fillna(0) * pd.to_numeric(df['unit_price'], errors='coerce').fillna(0)
    return df.dropna(subset=['sale_date'])


def monthly_series():
    df = sales_df()
    if df.empty:
        return pd.Series(dtype=float)
    series = df.set_index('sale_date').resample('MS')['revenue'].sum().sort_index()
    return series.asfreq('MS', fill_value=0)


def make_forecast(periods=6):
    series = monthly_series()
    if len(series) < 2:
        raise ValueError('Add at least two months of sales data before forecasting.')
    periods = max(1, min(int(periods), 24))
    # Prefer Holt-Winters seasonal forecasting when there is enough history; fall back safely.
    model_name = 'Holt trend'
    values = series.astype(float)
    forecast_values = None
    if len(values) >= 24:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            fit = ExponentialSmoothing(values, trend='add', seasonal='add', seasonal_periods=12, initialization_method='estimated').fit(optimized=True)
            forecast_values = fit.forecast(periods)
            model_name = 'Holt-Winters seasonal'
        except Exception:
            forecast_values = None
    if forecast_values is None:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            fit = ExponentialSmoothing(values, trend='add', seasonal=None, initialization_method='estimated').fit(optimized=True)
            forecast_values = fit.forecast(periods)
            model_name = 'Holt trend'
        except Exception:
            x = np.arange(len(values), dtype=float)
            slope, intercept = np.polyfit(x, values.values, 1)
            forecast_values = pd.Series([max(0, intercept + slope * (len(values) + i)) for i in range(periods)], index=pd.date_range(values.index[-1] + pd.offsets.MonthBegin(1), periods=periods, freq='MS'))
            model_name = 'Linear trend fallback'
    forecast_values = pd.Series(forecast_values).clip(lower=0)
    return series, forecast_values, model_name


@app.route('/')
def home():
    return render_template('index.html')

@app.get('/api/sales')
def get_sales():
    return jsonify(all_sales())

@app.post('/api/sales')
def add_sale():
    data = request.get_json(silent=True) or request.form
    try:
        sale_date = str(data.get('sale_date', '')).strip()
        product = str(data.get('product', '')).strip()
        category = str(data.get('category', 'General')).strip() or 'General'
        units = int(data.get('units', 0))
        unit_price = float(data.get('unit_price', 0))
        promotion = 1 if str(data.get('promotion', '0')).lower() in ('1','true','yes','on') else 0
        pd.to_datetime(sale_date, errors='raise')
        if not product or units < 1 or unit_price < 0:
            raise ValueError('Enter a product, units greater than zero, and a non-negative price.')
        with connect() as con:
            cur = con.execute('INSERT INTO sales (sale_date,product,category,units,unit_price,promotion) VALUES (?,?,?,?,?,?)', (sale_date, product, category, units, unit_price, promotion))
            sale_id = cur.lastrowid
        return jsonify({'ok': True, 'id': sale_id}), 201
    except Exception as e:
        return jsonify({'error': str(e) or 'Please check the sales fields.'}), 400

@app.put('/api/sales/<int:sale_id>')
def update_sale(sale_id):
    data = request.get_json(silent=True) or {}
    try:
        sale_date = str(data.get('sale_date', '')).strip(); product = str(data.get('product', '')).strip()
        category = str(data.get('category', 'General')).strip() or 'General'
        units = int(data.get('units', 0)); unit_price = float(data.get('unit_price', 0))
        promotion = 1 if str(data.get('promotion', '0')).lower() in ('1','true','yes','on') else 0
        pd.to_datetime(sale_date, errors='raise')
        if not product or units < 1 or unit_price < 0: raise ValueError('Please enter valid product, units, and price.')
        with connect() as con:
            cur = con.execute('UPDATE sales SET sale_date=?,product=?,category=?,units=?,unit_price=?,promotion=? WHERE id=?', (sale_date,product,category,units,unit_price,promotion,sale_id))
            if cur.rowcount == 0: return jsonify({'error':'Sale record not found.'}), 404
        return jsonify({'ok':True})
    except Exception as e: return jsonify({'error':str(e)}), 400

@app.delete('/api/sales/<int:sale_id>')
def delete_sale(sale_id):
    with connect() as con:
        cur = con.execute('DELETE FROM sales WHERE id=?', (sale_id,))
    if cur.rowcount == 0: return jsonify({'error':'Sale record not found.'}), 404
    return jsonify({'ok':True})

@app.get('/api/overview')
def overview():
    df = sales_df()
    if df.empty: return jsonify({'records':0,'revenue':0,'units':0,'avg_order':0,'labels':[],'monthly':[],'categories':[],'category_values':[]})
    monthly = df.set_index('sale_date').resample('MS')['revenue'].sum().sort_index()
    cat = df.groupby('category')['revenue'].sum().sort_values(ascending=False)
    return jsonify({'records':int(len(df)),'revenue':round(float(df.revenue.sum()),2),'units':int(df.units.sum()),'avg_order':round(float(df.revenue.mean()),2),'labels':[d.strftime('%b %Y') for d in monthly.index], 'monthly':[round(float(x),2) for x in monthly.values], 'categories':cat.index.tolist(),'category_values':[round(float(x),2) for x in cat.values]})

@app.get('/api/analysis')
def analysis():
    df = sales_df(); s = monthly_series()
    if df.empty or len(s) < 2: return jsonify({'error':'Add sales records across at least two months to analyze time-series patterns.'}), 400
    moving = s.rolling(3, min_periods=1).mean()
    change = ((s.iloc[-1] - s.iloc[-2]) / s.iloc[-2] * 100) if s.iloc[-2] else 0
    by_month = df.groupby(df.sale_date.dt.month)['revenue'].mean().reindex(range(1,13), fill_value=0)
    return jsonify({'labels':[d.strftime('%b %Y') for d in s.index], 'revenue':[round(float(x),2) for x in s.values], 'moving_average':[round(float(x),2) for x in moving.values], 'month_labels':['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], 'seasonality':[round(float(x),2) for x in by_month.values], 'latest_change_pct':round(float(change),2), 'periods':len(s), 'trend':'Increasing' if s.iloc[-1] > s.iloc[0] else ('Decreasing' if s.iloc[-1] < s.iloc[0] else 'Stable')})

@app.get('/api/forecast')
def forecast():
    try:
        periods = int(request.args.get('periods', 6))
        actual, pred, model_name = make_forecast(periods)
        # Back-test recent observations for useful model metrics.
        mae = rmse = mape = None
        if len(actual) >= 5:
            train = actual.iloc[:-3]; test = actual.iloc[-3:]
            try:
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                fitted = ExponentialSmoothing(train.astype(float), trend='add', seasonal=None, initialization_method='estimated').fit(optimized=True)
                back = np.maximum(0, fitted.forecast(len(test)).values)
                errors = test.values - back
                mae = float(np.mean(np.abs(errors))); rmse = float(np.sqrt(np.mean(errors**2)))
                nonzero = test.values != 0
                if nonzero.any(): mape = float(np.mean(np.abs(errors[nonzero] / test.values[nonzero])) * 100)
            except Exception: pass
        return jsonify({'model':model_name,'actual_labels':[d.strftime('%b %Y') for d in actual.index], 'actual':[round(float(x),2) for x in actual.values], 'forecast_labels':[d.strftime('%b %Y') for d in pred.index], 'forecast':[round(float(x),2) for x in pred.values], 'periods':periods, 'mae':round(mae,2) if mae is not None else None, 'rmse':round(rmse,2) if rmse is not None else None, 'mape':round(mape,2) if mape is not None else None, 'total_forecast':round(float(pred.sum()),2)})
    except Exception as e: return jsonify({'error':str(e)}), 400

@app.get('/api/insights')
def insights():
    df = sales_df(); s = monthly_series()
    if df.empty or len(s) < 2: return jsonify({'error':'Add sales data spanning at least two months to generate insights.'}), 400
    latest = float(s.iloc[-1]); previous = float(s.iloc[-2]); delta = ((latest-previous)/previous*100) if previous else 0
    top = df.groupby('product').revenue.sum().sort_values(ascending=False)
    promo = df[df.promotion == 1].revenue.mean() if (df.promotion == 1).any() else None
    no_promo = df[df.promotion == 0].revenue.mean() if (df.promotion == 0).any() else None
    notes = [f'Most recent month revenue is ₹{latest:,.2f}, compared with ₹{previous:,.2f} in the prior month ({delta:+.1f}%).', f"Overall trend is {'upward' if s.iloc[-1] > s.iloc[0] else 'downward' if s.iloc[-1] < s.iloc[0] else 'mostly flat'} across the available monthly history."]
    if len(top): notes.append(f"Top product by recorded revenue is {top.index[0]} (₹{top.iloc[0]:,.2f} total).")
    if promo is not None and no_promo is not None: notes.append('Average record revenue is higher on promotion records.' if promo > no_promo else 'Average record revenue is not higher on promotion records; review promotion timing and product mix.')
    return jsonify({'model':'Holt-Winters seasonal when 24+ monthly periods are available; otherwise Holt trend','records':int(len(df)),'months':int(len(s)),'trend':'Increasing' if s.iloc[-1] > s.iloc[0] else ('Decreasing' if s.iloc[-1] < s.iloc[0] else 'Stable'),'latest_change_pct':round(float(delta),2),'top_product':str(top.index[0]) if len(top) else 'N/A','top_product_revenue':round(float(top.iloc[0]),2) if len(top) else 0,'notes':notes,'limitations':['Forecasts assume historical patterns continue.','Promotional effects are descriptive unless a model explicitly includes promotion as an input.','More months of clean data generally improve seasonal analysis.']})

@app.post('/api/upload')
def upload_csv():
    f = request.files.get('file')
    if not f: return jsonify({'error':'Choose a CSV file first.'}), 400
    try:
        df = pd.read_csv(f)
        aliases = {c.lower().strip().replace(' ','_'):c for c in df.columns}
        def col(*names):
            for n in names:
                if n in aliases: return aliases[n]
            return None
        date_col = col('sale_date','date','order_date'); product_col = col('product','product_name','item'); units_col = col('units','quantity','qty'); price_col = col('unit_price','price','selling_price')
        if not all([date_col, product_col, units_col, price_col]):
            return jsonify({'error':'CSV needs columns: date (or sale_date), product, units (or quantity), and unit_price (or price). Optional: category, promotion.'}), 400
        cat_col = col('category'); promo_col = col('promotion','promotional','is_promotion')
        rows=[]
        for _, r in df.iterrows():
            dt = pd.to_datetime(r[date_col], errors='raise').strftime('%Y-%m-%d')
            product = str(r[product_col]).strip()
            units = int(r[units_col]); price = float(r[price_col])
            if not product or units < 1 or price < 0: continue
            cat = str(r[cat_col]).strip() if cat_col and pd.notna(r[cat_col]) else 'General'
            promo = 1 if promo_col and str(r[promo_col]).lower() in ('1','true','yes','on') else 0
            rows.append((dt,product,cat or 'General',units,price,promo))
        if not rows: return jsonify({'error':'No valid sales rows found in CSV.'}), 400
        with connect() as con: con.executemany('INSERT INTO sales (sale_date,product,category,units,unit_price,promotion) VALUES (?,?,?,?,?,?)', rows)
        return jsonify({'ok':True,'imported':len(rows)})
    except Exception as e: return jsonify({'error':f'Could not import CSV: {e}'}), 400

@app.get('/api/export')
def export_csv():
    df = sales_df().drop(columns=['revenue'], errors='ignore')
    out = io.StringIO(); df.to_csv(out, index=False)
    mem = io.BytesIO(out.getvalue().encode('utf-8')); mem.seek(0)
    return send_file(mem, mimetype='text/csv', as_attachment=True, download_name='sales_export.csv')

init_db()
if __name__ == '__main__':
    app.run(debug=True)
