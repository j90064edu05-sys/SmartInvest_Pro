import os
import sys
import json
import io
import base64
import time
from datetime import datetime, timedelta

# ==========================================
# 基礎環境自我診斷
# ==========================================
def diagnostic_check():
    """檢查 Python 路徑是否正常"""
    print(f">>> [診斷] Python 執行路徑: {sys.executable}")
    print(f">>> [診斷] Python 庫路徑: {sys.prefix}")

diagnostic_check()

# ==========================================
# 自動化套件安裝
# ==========================================
def install_and_import(package, import_name=None):
    import_name = import_name or package
    try:
        __import__(import_name)
    except ImportError:
        print(f">>> 偵測到缺少套件: {package}，正在嘗試自動安裝...")
        exit_code = os.system(f'"{sys.executable}" -m pip install {package} --no-cache-dir')
        if exit_code != 0:
            print(f">>> [嚴重錯誤] 無法自動安裝 {package}。")
            sys.exit(1)
        try:
            __import__(import_name)
        except ImportError:
            sys.exit(1)

install_and_import('pandas', 'pandas')
install_and_import('numpy', 'numpy')
install_and_import('yfinance', 'yfinance')
install_and_import('pandas_market_calendars', 'pandas_market_calendars')
install_and_import('matplotlib', 'matplotlib')
install_and_import('gdown', 'gdown')
install_and_import('google-api-python-client', 'googleapiclient')
install_and_import('google-auth-oauthlib', 'google_auth_oauthlib')

import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import gdown

# Email 相關模組
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

# 設定 Matplotlib 後端
plt.switch_backend('Agg') 

# ==========================================
# Google API 函式庫導入預檢
# ==========================================
HAS_GCP_LIBS = False
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    HAS_GCP_LIBS = True
except ImportError:
    HAS_GCP_LIBS = False

# ==========================================
# 中文字型自動設定
# ==========================================
font_filename = "taipei_sans_tc_beta.ttf"
if not os.path.exists(font_filename):
    print(">>> 偵測到缺少中文字型，正在嘗試下載...")
    url = "https://drive.google.com/uc?id=1eGAsTN1HBpJAkeVM57_C7ccp7hbgSz3_&export=download"
    try:
        gdown.download(url, font_filename, quiet=True)
    except: pass

if os.path.exists(font_filename):
    try:
        fm.fontManager.addfont(font_filename)
        font_prop = fm.FontProperties(fname=font_filename)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
    except: pass

# ==========================================
# 1. 雲端與通訊資源管理員
# ==========================================
class ResourceManager:
    folder_id = None
    drive_service = None
    gmail_service = None

    def __init__(self, folder_name="SmartInvest_Pro"):
        self.folder_id = None
        self.drive_service = None
        self.gmail_service = None
        self.folder_name = folder_name
        self.base_path = ""
        self.is_colab = self._detect_colab()
        self.is_github = os.environ.get('GITHUB_ACTIONS') == 'true'
        
        self.SCOPES = [
            'https://www.googleapis.com/auth/drive.file',
            'https://www.googleapis.com/auth/spreadsheets.readonly',
            'https://www.googleapis.com/auth/gmail.send'
        ]

        if self.is_colab:
            self._setup_colab_paths()
        else:
            self.base_path = os.getcwd()
            
        if HAS_GCP_LIBS:
            self._authenticate_services()

    def _detect_colab(self):
        try:
            import google.colab
            return True
        except: return False

    def _setup_colab_paths(self):
        from google.colab import drive
        if not os.path.exists('/content/drive'):
            drive.mount('/content/drive', force_remount=True)
        self.base_path = f'/content/drive/MyDrive/{self.folder_name}/'
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def _authenticate_services(self):
        creds = None
        token_path = os.path.join(self.base_path, 'token.json')
        cred_path = os.path.join(self.base_path, 'credentials.json')
        
        if os.path.exists(token_path):
            try: creds = Credentials.from_authorized_user_file(token_path, self.SCOPES)
            except: creds = None

        if creds and not all(s in (creds.scopes or []) for s in self.SCOPES):
            creds = None

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except: creds = None
            
            if not creds:
                if self.is_github:
                    print(">>> [系統] GitHub Actions 模式：Token 失效，跳過雲端授權。")
                    return 
                if not os.path.exists(cred_path):
                    print(">>> [系統] 找不到憑證檔案，使用純本地模式。")
                    return
                flow = InstalledAppFlow.from_client_secrets_file(cred_path, self.SCOPES)
                if self.is_colab:
                    flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
                    auth_url, _ = flow.authorization_url(prompt='consent')
                    print(f"\n授權連結: {auth_url}")
                    code = input("請輸入驗證碼: ")
                    flow.fetch_token(code=code)
                    creds = flow.credentials
                else:
                    creds = flow.run_local_server(port=0)
            
            if creds:
                with open(token_path, 'w') as token:
                    token.write(creds.to_json())
        
        try:
            self.drive_service = build('drive', 'v3', credentials=creds)
            self.gmail_service = build('gmail', 'v1', credentials=creds)
            self._ensure_folder_exists()
        except Exception as e:
            print(f">>> [警告] 服務初始化受限: {e}")

    def _ensure_folder_exists(self):
        if not self.drive_service: return
        try:
            query = f"name = '{self.folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            res = self.drive_service.files().list(q=query).execute().get('files', [])
            if not res:
                meta = {'name': self.folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
                file = self.drive_service.files().create(body=meta, fields='id').execute()
                self.folder_id = file.get('id')
            else:
                self.folder_id = res[0]['id']
        except: self.folder_id = None

    def load_local_config(self, filename="config.json"):
        local_path = os.path.join(self.base_path, filename)
        if os.path.exists(local_path):
            try:
                with open(local_path, 'r', encoding='utf-8') as f:
                    print(f">>> [系統] 從本地環境載入設定: {filename}")
                    return json.load(f)
            except: pass
        
        fid = getattr(self, 'folder_id', None)
        if self.drive_service and fid:
            try:
                query = f"name = '{filename}' and '{fid}' in parents and trashed = false"
                res = self.drive_service.files().list(q=query).execute().get('files', [])
                if res:
                    request = self.drive_service.files().get_media(fileId=res[0]['id'])
                    fh = io.BytesIO()
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done: _, done = downloader.next_chunk()
                    print(f">>> [雲端] 從 Drive 載入設定: {filename}")
                    return json.loads(fh.getvalue().decode('utf-8'))
            except: pass
        return None

    def save_file_to_drive(self, filename, data):
        content = ""
        mimetype = 'application/json'
        if isinstance(data, pd.DataFrame): 
            content = data.to_csv(index=False, encoding='utf-8-sig')
            mimetype = 'text/csv'
        else:
            content = json.dumps(data, indent=4, ensure_ascii=False)

        local_path = os.path.join(self.base_path, filename)
        try:
            with open(local_path, 'w', encoding='utf-8') as f: f.write(content)
        except: pass

        fid = getattr(self, 'folder_id', None)
        if self.drive_service and fid:
            try:
                with open("temp.tmp", "w", encoding="utf-8") as f: f.write(content)
                media = MediaFileUpload("temp.tmp", mimetype=mimetype)
                query = f"name = '{filename}' and '{fid}' in parents and trashed = false"
                res = self.drive_service.files().list(q=query).execute().get('files', [])
                if res:
                    self.drive_service.files().update(fileId=res[0]['id'], media_body=media).execute()
                else:
                    meta = {'name': filename, 'parents': [fid]}
                    self.drive_service.files().create(body=meta, media_body=media).execute()
                os.remove("temp.tmp")
            except: pass

    def read_web_csv(self, url):
        if not url or "http" not in url: return None
        try:
            print(f">>> 讀取 Web CSV...")
            df = pd.read_csv(url)
            return df
        except: return None

    def send_email_with_chart(self, to, subject, body_html, image_bytes=None):
        if not self.gmail_service or not to: return
        try:
            msg = MIMEMultipart('related')
            msg['to'] = to
            msg['subject'] = subject
            msg_alt = MIMEMultipart('alternative')
            msg.attach(msg_alt)
            msg_text = MIMEText(body_html, 'html')
            msg_alt.attach(msg_text)
            if image_bytes:
                img = MIMEImage(image_bytes.getvalue())
                img.add_header('Content-ID', '<portfolio_chart>')
                img.add_header('Content-Disposition', 'inline')
                msg.attach(img)
            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode('utf-8')
            self.gmail_service.users().messages().send(userId='me', body={'raw': raw}).execute()
            print(f">>> [通知] 郵件發送成功: {to}")
        except Exception as e: print(f">>> [錯誤] 發信失敗: {e}")

# ==========================================
# 2. 智投雙軌系統
# ==========================================
class HybridInvestSystem:
    def __init__(self):
        self.rm = ResourceManager()
        self.xtai = mcal.get_calendar('XTAI')
        self.config = self._init_config()
        self.rate_cache = {} # 匯率快取

    def _init_config(self):
        default_conf = {
            "transaction_csv_url": "", 
            "backtest_start_date": "2020-01-01",
            "monthly_budget": 20000,
            "cash_pool_ratio": 0.1,
            "fee_discount": 1,
            "email_config": {"enable": True, "receiver_email": ""},
            "targets": {
                "00808.TW": {"ratio": 0.3, "mode": "TECH", "name": "華南永昌優選50"},
                "AAPL": {"ratio": 0.3, "mode": "PYRAMID", "name": "Apple Inc."},
                "USD-TD": {"ratio": 0.1, "mode": "ACTIVE", "name": "美金定存"}
            },
            "pyramid_levels": {
                "S1": {"drop": -0.15, "mult": 1.0}, "S2": {"drop": -0.25, "mult": 1.5}, "S3": {"drop": -0.35, "mult": 2.0}
            }
        }
        conf = self.rm.load_local_config() or default_conf
        for k, v in default_conf.items():
            if k not in conf: conf[k] = v
        if not os.environ.get('GITHUB_ACTIONS'):
            self.rm.save_file_to_drive("config.json", conf)
        return conf

    def get_currency_and_rate(self, ticker, is_fixed=False):
        """
        判斷幣別並獲取對台幣匯率
        Returns: (currency_code, rate_to_twd)
        """
        # 1. 判斷幣別代碼
        if str(ticker).endswith('.TW') or str(ticker).endswith('.TWO') or str(ticker) == 'TWD':
            return 'TWD', 1.0
        
        currency = 'USD' # 預設美金
        
        if is_fixed:
            # 定存類：移除 -TD 後綴 (e.g. USD-TD -> USD)
            if str(ticker).endswith('-TD'):
                currency = str(ticker).replace('-TD', '')
            elif str(ticker) in ['TWD-TD']: # 特例
                return 'TWD', 1.0
        else:
            # 股票類：若無 .TW 則視為美股/外幣
            currency = 'USD'

        # 2. 獲取匯率 (使用快取)
        if currency == 'TWD': return 'TWD', 1.0
        
        if currency in self.rate_cache:
            return currency, self.rate_cache[currency]
        
        # Yahoo Finance 匯率代碼規則: USD -> TWD=X, JPY -> JPYTWD=X
        yahoo_symbol = f"TWD=X" if currency == 'USD' else f"{currency}TWD=X"
        try:
            rate_data = yf.download(yahoo_symbol, period="1d", progress=False)
            if not rate_data.empty:
                rate = float(rate_data['Close'].iloc[-1])
                self.rate_cache[currency] = rate
                print(f">>> [匯率更新] 1 {currency} = {rate:.4f} TWD")
                return currency, rate
        except Exception as e:
            print(f">>> [警告] 無法取得 {currency} 匯率: {e}")
        
        return currency, 1.0 # 失敗時回傳1

    def calculate_ema_talib(self, series, span):
        values = series.values
        if len(values) < span: return pd.Series(np.nan, index=series.index)
        sma_seed = np.mean(values[:span])
        ema_result = np.full(len(values), np.nan)
        ema_result[span - 1] = sma_seed
        k = 2 / (span + 1)
        for i in range(span, len(values)):
            ema_result[i] = (values[i] - ema_result[i-1]) * k + ema_result[i-1]
        return pd.Series(ema_result, index=series.index)

    def calculate_indicators(self, df):
        df = df.sort_index()
        price = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
        
        ema12 = self.calculate_ema_talib(price, 12)
        ema26 = self.calculate_ema_talib(price, 26)
        df['DIF'] = ema12 - ema26
        df['DEA'] = self.calculate_ema_talib(df['DIF'].dropna(), 9)
        df['DEA'] = df['DEA'].reindex(df.index)
        df['OSC'] = df['DIF'] - df['DEA']
        
        low9 = df['Low'].rolling(window=9).min()
        high9 = df['High'].rolling(window=9).max()
        rsv = (price - low9) / (high9 - low9) * 100
        df['K'] = rsv.ewm(com=2, adjust=False).mean()
        df['D'] = df['K'].ewm(com=2, adjust=False).mean()
        
        df_w = df.resample('W-FRI').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'})
        w_low = df_w['Low'].rolling(window=9).min()
        w_high = df_w['High'].rolling(window=9).max()
        w_rsv = (df_w['Close'] - w_low) / (w_high - w_low) * 100
        df_w['WK'] = w_rsv.ewm(com=2, adjust=False).mean()
        df = df.join(df_w[['WK']], how='left').ffill()
        
        for m in [20, 60, 120]: df[f'MA{m}'] = price.rolling(window=m).mean()
        return df

    def evaluate_strategy_today(self, ticker, df, war_chest, portfolio_status, rate):
        if ticker not in self.config["targets"]: return "不適用", 0
        t_conf = self.config["targets"][ticker]
        budget = self.config["monthly_budget"]
        base_alloc = budget * t_conf["ratio"]
        today = df.index[-1]
        last = df.iloc[-1]
        prev = df.iloc[-2]
        idx = df.index.get_loc(today)
        
        suggestion = "觀望"
        invest_amt = 0
        
        # 基礎投資
        month_base_invested = portfolio_status.get("month_base_invested", 0)
        remaining_base = max(0, base_alloc - month_base_invested)
        
        if remaining_base > 100:
            sched = self.xtai.schedule(start_date=today, end_date=today + timedelta(days=10))
            is_last_day = (sched.index[0].month != sched.index[1].month) if len(sched) > 1 else True
            
            has_dc = False
            for lb in range(1, 4):
                if idx-lb<0: continue
                if df.iloc[idx-lb]['DIF'] > df.iloc[idx-lb]['DEA'] and df.iloc[idx-lb+1]['DIF'] < df.iloc[idx-lb+1]['DEA']:
                    has_dc = True; break
            
            tech_trigger = (has_dc and last['Close'] < prev['Close'] and last['OSC'] < 0)
            if tech_trigger:
                suggestion = "建議基礎投資 (技術)"; invest_amt += remaining_base
            elif is_last_day:
                suggestion = "建議基礎投資 (保底)"; invest_amt += remaining_base
            elif month_base_invested >= base_alloc * 0.9:
                suggestion = "觀望 (基礎額滿)"
            else:
                reasons = []
                if not has_dc: reasons.append("無死叉")
                if not (last['Close'] < prev['Close']): reasons.append("未收跌")
                if not (last['OSC'] < 0): reasons.append("OSC正")
                suggestion = f"觀望 ({'/'.join(reasons)})"

        # 加碼投資
        executed_extra = portfolio_status.get("executed_extra", [])
        # 平均成本需還原回原幣別 (因為技術分析用原幣)
        # avg_cost_twd = portfolio_status.get("avg_cost", 0) 
        # 但這裡的 df 也是原幣，所以我們要用原幣成本比較
        # 修正: portfolio_status 傳入時應為 TWD，這裡除以匯率還原
        avg_cost_original = portfolio_status.get("avg_cost", 0) / rate if rate > 0 else 0
        
        mode = t_conf["mode"]
        extra_amt = 0
        extra_reason = ""
        
        if mode == "PYRAMID" and avg_cost_original > 0:
            drop = (last['Close'] - avg_cost_original) / avg_cost_original
            for s_name, s_cfg in self.config["pyramid_levels"].items():
                if drop <= s_cfg["drop"] and s_name not in executed_extra:
                    req_amt = base_alloc * s_cfg["mult"]
                    if war_chest >= req_amt:
                        extra_amt = req_amt
                        extra_reason = f"加碼({s_name})"
                    else:
                        extra_reason = f"加碼({s_name})但資金不足"
                    break
        elif mode == "TECH":
            triggered = False
            req_amt = base_alloc
            if (last['K'] < 20 or last['WK'] < 20) and "K_OVER" not in executed_extra:
                if war_chest >= req_amt:
                    extra_amt = req_amt
                    extra_reason = "K值加碼"
                else:
                    extra_reason = "K值加碼(資金不足)"
                triggered = True
            if not triggered:
                for ma in ['MA60', 'MA120']:
                    mv = last[ma]
                    if last['Close'] >= mv and (last['Close']-mv)/mv < 0.02 and (last['Low'] <= mv or prev['Low'] <= mv):
                        if ma not in executed_extra:
                            if war_chest >= req_amt:
                                extra_amt = req_amt
                                extra_reason = f"{ma}加碼"
                            else:
                                extra_reason = f"{ma}加碼(資金不足)"
                            break
        
        if extra_amt > 0:
            invest_amt += extra_amt
            if "建議" in suggestion: suggestion += f" & {extra_reason}"
            else: suggestion = f"建議{extra_reason}"
        elif "資金不足" in extra_reason:
            suggestion += f" & {extra_reason}"

        return suggestion, invest_amt

    def run_backtest(self):
        start_date = self.config["backtest_start_date"]
        tickers = list(self.config["targets"].keys())
        data_map = {}
        warmup_date = (pd.to_datetime(start_date) - timedelta(days=365*3)).strftime('%Y-%m-%d')
        for t in tickers:
            raw = yf.download(t, period="max", progress=False)
            if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
            data_map[t] = self.calculate_indicators(raw).loc[start_date:]
        history = []
        portfolio = {t: {"shares": 0, "cost": 0} for t in tickers}
        war_chest = 0
        budget = self.config["monthly_budget"]
        all_dates = data_map[tickers[0]].index
        current_month = -1
        month_base_done = {t: False for t in tickers}
        extra_done = {t: set() for t in tickers}
        for date in all_dates:
            if date.month != current_month:
                current_month = date.month
                war_chest += budget * self.config["cash_pool_ratio"]
                month_base_done = {t: False for t in tickers}
                extra_done = {t: set() for t in tickers}
            for t in tickers:
                df = data_map[t]
                if date not in df.index: continue
                price = df.loc[date, 'Close']
                t_conf = self.config["targets"][t]
                base_budget = budget * t_conf["ratio"]
                # 簡化：回測暫不模擬多幣別匯率波動，皆視為 TWD
                if not month_base_done[t]:
                    is_last = (date.month != (date + timedelta(days=5)).month)
                    idx = df.index.get_loc(date)
                    has_dc = False
                    for i in range(3):
                        check_idx = idx - i
                        if check_idx <= 0: continue
                        if df.iloc[check_idx-1]['DIF'] > df.iloc[check_idx-1]['DEA'] and \
                           df.iloc[check_idx]['DIF'] < df.iloc[check_idx]['DEA']:
                            has_dc = True; break
                    tech = (has_dc and price < df.iloc[idx-1]['Close'] and df.loc[date, 'OSC'] < 0)
                    if tech or is_last:
                        sh = base_budget / price
                        portfolio[t]["shares"] += sh
                        portfolio[t]["cost"] += base_budget
                        month_base_done[t] = True
                        history.append({"日期": date.strftime('%Y-%m-%d'), "標的": t, "策略": "基礎投資", "金額": int(base_budget), "股數": round(sh, 2), "成交價": round(price, 2)})
        return pd.DataFrame(history), war_chest

    def generate_chart(self, summary_df, cash_tickers):
        try:
            plot_df = summary_df[~summary_df['標的'].isin(cash_tickers)].copy()
            if plot_df.empty: return None
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            x = np.arange(len(plot_df))
            width = 0.35
            axes[0].bar(x - width/2, plot_df['金額'], width, label='投入成本', color='#95a5a6')
            axes[0].bar(x + width/2, plot_df['MarketValue'], width, label='目前市值', color='#e74c3c')
            axes[0].set_ylabel('金額 (TWD)')
            axes[0].set_title('投資組合: 成本 vs 市值')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(plot_df['標的'], rotation=45)
            axes[0].legend()
            axes[1].pie(plot_df['MarketValue'], labels=plot_df['標的'], autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
            axes[1].set_title('資產市值配置比例')
            plt.tight_layout()
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', dpi=100)
            img_buf.seek(0)
            plt.close()
            return img_buf
        except: return None

    def analyze_and_notify(self, df, mode="REAL", war_chest_sim=0):
        print(f">>> 執行分析報告 [{mode}]...")
        df.columns = [c.strip() for c in df.columns]
        war_chest_mv = 0
        cash_tickers = set()

        if mode == "REAL":
            type_cols = [c for c in ['策略', '類別', '類型'] if c in df.columns]
            price_col = next((c for c in ['價格', '成交價', 'Price'] if c in df.columns), None)
            is_fixed = pd.Series(False, index=df.index)
            if type_cols:
                for col in type_cols:
                    is_fixed |= df[col].astype(str).str.contains('定存', na=False)
            
            # 定存邏輯：市值 = 股數(外幣) * 匯率(價格欄位)
            # 若為台幣定存(TWD-TD)，價格=1
            if price_col:
                df['RowMV'] = 0.0
                df.loc[is_fixed, 'RowMV'] = df.loc[is_fixed, '股數'] * df.loc[is_fixed, price_col]
                war_chest_mv = df.loc[is_fixed, 'RowMV'].sum()
            else:
                war_chest_mv = df.loc[is_fixed, '金額'].sum()
            
            cash_tickers = set(df[is_fixed]['標的'].unique())
            cash_tickers.add('定存'); cash_tickers.add('CASH')
        else:
            war_chest_mv = war_chest_sim
            cash_tickers = {'定存', 'CASH'}

        aggs = {"金額": "sum", "股數": "sum"}
        if mode == "REAL" and 'RowMV' in df.columns: aggs["RowMV"] = "sum"
        summary = df.groupby("標的").agg(aggs).reset_index()
        summary['MarketValue'] = 0.0
        summary['Price'] = 0.0
        now = datetime.now()
        
        conf_html = f"""<div style='background-color:#f9f9f9; padding:10px; margin-bottom:15px; border-radius:5px; font-size:13px; color:#555;'>
            <b>📊 系統參數：</b><br>• 每月預算: {self.config['monthly_budget']:,.0f}<br>• 策略配置:<br>"""
        for t, c in self.config["targets"].items():
            conf_html += f"&nbsp;&nbsp;- {t}: {c['ratio']*100:.0f}% ({c['mode']})<br>"
        conf_html += "</div>"

        for idx, row in summary.iterrows():
            ticker = row['標的']
            # 1. 判斷幣別 & 獲取匯率
            is_fd = ticker in cash_tickers or ticker.upper() == 'CASH'
            currency, rate = self.get_currency_and_rate(ticker, is_fixed=is_fd)
            
            # 2. 定存計算
            if is_fd:
                if mode == "REAL" and 'RowMV' in summary.columns:
                    summary.at[idx, 'MarketValue'] = row['RowMV']
                else:
                    summary.at[idx, 'MarketValue'] = war_chest_sim
                summary.at[idx, 'Price'] = rate # 顯示匯率作為定存價格
            
            # 3. 股票計算 (美股需乘匯率)
            else:
                try:
                    curr_data = yf.download(ticker, period="max", progress=False)
                    if isinstance(curr_data.columns, pd.MultiIndex):
                        curr_data.columns = curr_data.columns.get_level_values(0)
                    # 原幣價格
                    raw_price = float(curr_data['Close'].iloc[-1]) if not curr_data.empty else 0
                    
                    # 換算台幣
                    summary.at[idx, 'Price'] = raw_price * rate
                    summary.at[idx, 'MarketValue'] = row['股數'] * raw_price * rate
                except: 
                    summary.at[idx, 'Price'] = 0

        chart_bytes = self.generate_chart(summary, cash_tickers)
        html = f"<h2>智投報告 [{mode}] - {now.strftime('%Y-%m-%d')}</h2>{conf_html}"
        html += '<img src="cid:portfolio_chart" alt="Portfolio Chart" style="max-width:100%;"><br><hr>'
        html += "<table border='1' cellpadding='5' style='border-collapse:collapse; width:100%; font-family: Arial; font-size: 13px;'>"
        html += "<tr style='background:#f2f2f2;'><th>標的</th><th>股數</th><th>市值(TWD)</th><th>報酬率</th><th>DIF</th><th>OSC</th><th>K(日/週)</th>"
        if mode == "REAL": html += "<th>今日建議</th><th>金額</th><th>股數</th>"
        html += "</tr>"

        all_targets = sorted(list(set(list(summary['標的']) + list(self.config["targets"].keys()))))
        total_cost = 0; total_mv = 0
        fee_discount = self.config.get("fee_discount", 1.0)

        for ticker in all_targets:
            if ticker == 'CASH': continue
            row = summary[summary['標的'] == ticker]
            cost = row.iloc[0]['金額'] if not row.empty else 0
            shares = row.iloc[0]['股數'] if not row.empty else 0
            mv = row.iloc[0]['MarketValue'] if not row.empty else 0
            roi = (mv - cost) / cost * 100 if cost > 0 else 0
            total_cost += cost; total_mv += mv
            
            is_fd = ticker in cash_tickers
            suggestion = "-"; sugg_amt_str = "-"; sugg_shares_str = "-"; dif_d = "-"; osc_d = "-"; k_d = "-"
            
            # 淨損益
            tax_rate = 0.0
            if is_fd: tax_rate = 0.0
            elif str(ticker).startswith("00") or "ETF" in str(ticker):
                tax_rate = 0.0 if ("債" in str(ticker) or "B" in str(ticker)) else 0.001
            else: tax_rate = 0.003
            
            # 手續費 & 稅 (用台幣市值算)
            handling_fee = 0 if is_fd else (mv * 0.001425 * fee_discount)
            trans_tax = mv * tax_rate
            net_profit = (mv - cost) - handling_fee - trans_tax
            net_roi = (net_profit / cost * 100) if cost > 0 else 0

            if mode == "REAL" and not is_fd:
                try:
                    # 取得匯率
                    curr, rate = self.get_currency_and_rate(ticker)
                    hist_data = yf.download(ticker, period="max", progress=False)
                    if isinstance(hist_data.columns, pd.MultiIndex): hist_data.columns = hist_data.columns.get_level_values(0)
                    if not hist_data.empty:
                        raw_price = float(hist_data['Close'].iloc[-1])
                        # 技術指標用原幣算
                        hist_data = self.calculate_indicators(hist_data)
                        last = hist_data.iloc[-1]
                        
                        dif_val = last['DIF']; osc_val = last['OSC']
                        dif_d = f"<span style='color:{'red' if dif_val>0 else 'green'}'>{dif_val:.4f}</span>"
                        osc_d = f"<span style='color:{'red' if osc_val>0 else 'green'}'>{osc_val:.4f}</span>"
                        k_d = f"{last['K']:.0f} / {last['WK']:.0f}"

                        col_type = next((c for c in ['策略', '類別', '類型'] if c in df.columns), None)
                        month_base = 0; extra_types = []
                        if col_type:
                             df['日期'] = pd.to_datetime(df['日期'])
                             m_df = df[(df['標的'] == ticker) & (df['日期'].dt.month == now.month) & (df['日期'].dt.year == now.year)]
                             month_base = m_df[m_df[col_type].astype(str).str.contains('基礎', na=False)]['金額'].sum()
                             for t in m_df[col_type].unique():
                                 t_str = str(t)
                                 if "金字塔" in t_str: extra_types.append(t_str.split('_')[-1])
                                 if "技術" in t_str or "K值" in t_str: extra_types.append("K_OVER" if "K值" in t_str else t_str)
                                 if "加碼" in t_str and "MA" in t_str: extra_types.append(t_str) 

                        # 策略運算 (傳入台幣成本, 內部除以匯率還原)
                        sugg_text, sugg_val_twd = self.evaluate_strategy_today(ticker, hist_data, war_chest_mv, {"month_base_invested": month_base, "executed_extra": extra_types, "avg_cost": cost/shares if shares>0 else 0}, rate)
                        suggestion = sugg_text
                        if sugg_val_twd > 0:
                            sugg_amt_str = f"{sugg_val_twd:,.0f}"
                            # 建議股數 = 建議金額(TWD) / 匯率 / 原幣股價
                            est_shares = int(sugg_val_twd / rate / raw_price)
                            sugg_shares_str = f"{est_shares}"
                        
                        print(f"{ticker:<10} {shares:>8.0f} {mv:>10,.0f} {roi:>7.2f}% (淨{net_roi:.2f}%) DIF:{dif_val:.4f} OSC:{osc_val:.4f} {suggestion:<10}")
                except: pass
            
            roi_color = "red" if roi > 0 else "green"
            html += f"<tr><td>{ticker}</td><td>{shares:,.0f}</td><td>{mv:,.0f}</td><td style='color:{roi_color}'>{roi:+.2f}%</td>"
            html += f"<td align='center'>{dif_d}</td><td align='center'>{osc_d}</td><td align='center'>{k_d}</td>"
            if mode == "REAL":
                if is_fd: html += "<td align='center'>-</td><td align='center'>-</td><td align='center'>-</td>"
                else: html += f"<td>{suggestion}</td><td>{sugg_amt_str}</td><td>{sugg_shares_str}</td>"
            html += "</tr>"
        
        cash_cost_total = 0
        if mode == "REAL":
            type_cols = [c for c in ['策略', '類別', '類型'] if c in df.columns]
            is_fixed_agg = pd.Series(False, index=df.index)
            for col in type_cols: is_fixed_agg |= df[col].astype(str).str.contains('定存', na=False)
            cash_cost_total = df.loc[is_fixed_agg, '金額'].sum()
        else: cash_cost_total = war_chest_sim

        total_mv += war_chest_mv
        total_cost += cash_cost_total
        tot_roi = (total_mv - total_cost) / total_cost * 100 if total_cost > 0 else 0
        html += f"</table><br><b>總投入成本:</b> {total_cost:,.0f} TWD<br>"
        html += f"<b>總資產市值:</b> {total_mv:,.0f} TWD (帳面: <span style='color:{'red' if tot_roi>0 else 'green'}'>{tot_roi:+.2f}%</span>)<br>"
        wc_display = war_chest_mv if mode == "REAL" else war_chest_sim
        html += f"<b>加碼金餘額 (定存市值):</b> <span style='color:blue'>{wc_display:,.0f} TWD</span><br>"
        
        email_conf = self.config.get("email_config", {})
        if email_conf.get("enable") and email_conf.get("receiver_email"):
            self.rm.send_email_with_chart(email_conf["receiver_email"], f"智投報告 - {now.strftime('%Y-%m-%d')}", html, chart_bytes)

    def run(self):
        today_str = datetime.now().strftime('%Y-%m-%d')
        valid_days = self.xtai.valid_days(start_date=today_str, end_date=today_str)
        if len(valid_days) == 0:
            print(f">>> [休市通知] 今日 {today_str} 休市，自動跳過。")
            return
        url = self.config.get("transaction_csv_url", "")
        web_df = self.rm.read_web_csv(url)
        if web_df is not None and not web_df.empty:
            self.analyze_and_notify(web_df, mode="REAL")
        else:
            backtest_df, chest = self.run_backtest()
            self.analyze_and_notify(backtest_df, mode="BACKTEST", war_chest_sim=chest)

if __name__ == "__main__":
    app = HybridInvestSystem()
    app.run()