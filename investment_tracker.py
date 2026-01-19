import os
import json
import io
import base64
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Email 相關模組
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

# ==========================================
# 自動化套件安裝與匯入
# ==========================================
def install_and_import(package, import_name=None):
    import_name = import_name or package
    try:
        __import__(import_name)
    except ImportError:
        os.system(f'pip install {package}')
        __import__(import_name)

install_and_import('pandas_market_calendars')
install_and_import('yfinance')
install_and_import('matplotlib')
install_and_import('gdown')

import pandas_market_calendars as mcal
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import gdown

# 設定 Matplotlib 後端
plt.switch_backend('Agg') 

# ==========================================
# 中文字型自動設定
# ==========================================
font_filename = "taipei_sans_tc_beta.ttf"
if not os.path.exists(font_filename):
    print(">>> 偵測到缺少中文字型，正在下載台北思源黑體...")
    url = "https://drive.google.com/uc?id=1eGAsTN1HBpJAkeVM57_C7ccp7hbgSz3_&export=download"
    try:
        gdown.download(url, font_filename, quiet=True)
    except:
        pass

if os.path.exists(font_filename):
    try:
        fm.fontManager.addfont(font_filename)
        font_prop = fm.FontProperties(fname=font_filename)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

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
# 1. 雲端與通訊資源管理員 (ResourceManager)
# ==========================================
class ResourceManager:
    def __init__(self, folder_name="SmartInvest_Pro"):
        # [核心修復] 在最開頭明確初始化所有屬性，絕對防止 AttributeError
        self.folder_id = None
        self.drive_service = None
        self.gmail_service = None
        self.folder_name = folder_name
        self.base_path = ""
        self.is_colab = self._detect_colab()
        
        self.SCOPES = [
            'https://www.googleapis.com/auth/drive.file',
            'https://www.googleapis.com/auth/spreadsheets.readonly',
            'https://www.googleapis.com/auth/gmail.send'
        ]

        if self.is_colab:
            self._setup_colab_paths()
        else:
            # GitHub Actions 下 base_path 會是當前執行目錄
            self.base_path = os.getcwd()
            
        if HAS_GCP_LIBS:
            self._authenticate_services()

    def _detect_colab(self):
        try:
            import google.colab
            return True
        except:
            return False

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
        
        # 嘗試從本地讀取 token.json (GitHub Secret 還原出的檔案)
        if os.path.exists(token_path):
            try:
                creds = Credentials.from_authorized_user_file(token_path, self.SCOPES)
            except:
                creds = None

        # 檢查 Token 有效性與自動刷新
        if creds and not all(s in creds.scopes for s in self.SCOPES):
            creds = None

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except:
                    creds = None
            
            if not creds:
                if not os.path.exists(cred_path):
                    print(">>> [提示] 本地無憑證檔，切換至無雲端功能模式執行。")
                    return

                # GitHub Actions 是非互動環境，無法執行手動授權流程
                if os.environ.get('GITHUB_ACTIONS'):
                    print(">>> [錯誤] GitHub Actions 環境下 Token 已過期且無法手動授權，請更新 Repository Secret 中的 GCP_TOKEN_BASE64。")
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
            
            with open(token_path, 'w') as token:
                token.write(creds.to_json())
        
        # 建立服務客戶端
        try:
            self.drive_service = build('drive', 'v3', credentials=creds)
            self.gmail_service = build('gmail', 'v1', credentials=creds)
            # 在服務成功建立後才獲取資料夾 ID
            self._ensure_folder_exists()
        except Exception as e:
            print(f">>> [警告] Google 服務授權不完全 (Drive 存取受限): {e}")

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
        except:
            self.folder_id = None

    def load_local_config(self, filename="config.json"):
        # 1. 優先檢查本地檔案 (這對應到 GitHub Secret 直接還原出來的 config.json)
        local_path = os.path.join(self.base_path, filename)
        if os.path.exists(local_path):
            try:
                with open(local_path, 'r', encoding='utf-8') as f:
                    print(f">>> [系統] 成功從本地載入設定檔: {filename}")
                    return json.load(f)
            except Exception as e:
                print(f">>> [警告] 本地設定檔讀取失敗: {e}")
        
        # 2. 只有在本地找不到時，才嘗試從雲端讀取
        if self.drive_service and getattr(self, 'folder_id', None):
            try:
                query = f"name = '{filename}' and '{self.folder_id}' in parents and trashed = false"
                res = self.drive_service.files().list(q=query).execute().get('files', [])
                if res:
                    request = self.drive_service.files().get_media(fileId=res[0]['id'])
                    fh = io.BytesIO()
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done: _, done = downloader.next_chunk()
                    print(f">>> [雲端] 從 Google Drive 載入設定檔: {filename}")
                    return json.loads(fh.getvalue().decode('utf-8'))
            except:
                pass
        return None

    def save_file_to_drive(self, filename, data):
        content = ""
        mimetype = 'application/json'
        if isinstance(data, pd.DataFrame): 
            content = data.to_csv(index=False, encoding='utf-8-sig')
            mimetype = 'text/csv'
        else:
            content = json.dumps(data, indent=4, ensure_ascii=False)

        # 儲存到本地環境
        local_path = os.path.join(self.base_path, filename)
        try:
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except:
            pass

        # 雲端同步 (僅在服務可用且 folder_id 存在時)
        if self.drive_service and getattr(self, 'folder_id', None):
            try:
                with open("temp.tmp", "w", encoding="utf-8") as f: f.write(content)
                media = MediaFileUpload("temp.tmp", mimetype=mimetype)
                query = f"name = '{filename}' and '{self.folder_id}' in parents and trashed = false"
                res = self.drive_service.files().list(q=query).execute().get('files', [])
                if res:
                    self.drive_service.files().update(fileId=res[0]['id'], media_body=media).execute()
                else:
                    meta = {'name': filename, 'parents': [self.folder_id]}
                    self.drive_service.files().create(body=meta, media_body=media).execute()
                os.remove("temp.tmp")
            except:
                pass

    def read_web_csv(self, url):
        if not url or "http" not in url: return None
        try:
            print(f">>> 讀取交易資料...")
            df = pd.read_csv(url)
            return df
        except:
            return None

    def send_email_with_chart(self, to, subject, body_html, image_bytes=None):
        if not self.gmail_service or not to:
            print(">>> [警告] Gmail 服務未啟動或未設定收件人，跳過發信。")
            return
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
            print(f">>> [通知] 郵件已發送至: {to}")
        except Exception as e:
            print(f">>> [錯誤] 郵件發送失敗: {e}")

# ==========================================
# 2. 智投雙軌系統
# ==========================================
class HybridInvestSystem:
    def __init__(self):
        self.rm = ResourceManager()
        self.xtai = mcal.get_calendar('XTAI')
        self.config = self._init_config()

    def _init_config(self):
        default_conf = {
            "transaction_csv_url": "", 
            "backtest_start_date": "2020-01-01",
            "monthly_budget": 20000,
            "cash_pool_ratio": 0.1,
            "email_config": {"enable": True, "receiver_email": ""},
            "targets": {
                "00808.TW": {"ratio": 0.3, "mode": "TECH", "name": "華南永昌優選50"},
                "00895.TW": {"ratio": 0.3, "mode": "PYRAMID", "name": "富邦智慧車"},
                "00679B.TWO": {"ratio": 0.3, "mode": "ACTIVE", "name": "元大美債20年"}
            },
            "pyramid_levels": {
                "S1": {"drop": -0.15, "mult": 1.0},
                "S2": {"drop": -0.25, "mult": 1.5},
                "S3": {"drop": -0.35, "mult": 2.0}
            }
        }
        conf = self.rm.load_local_config() or default_conf
        for k, v in default_conf.items():
            if k not in conf: conf[k] = v
        
        # GitHub Actions 環境下避免反向同步到雲端 Drive，以保持單向安全性
        if not os.environ.get('GITHUB_ACTIONS'):
            self.rm.save_file_to_drive("config.json", conf)
        return conf

    def calculate_indicators(self, df):
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['DIF'] = ema12 - ema26
        df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
        df['OSC'] = df['DIF'] - df['DEA']
        
        low9 = df['Low'].rolling(window=9).min()
        high9 = df['High'].rolling(window=9).max()
        rsv = (df['Close'] - low9) / (high9 - low9) * 100
        df['K'] = rsv.ewm(com=2, adjust=False).mean()
        
        df_w = df.resample('W-FRI').last()
        low9w = df_w['Low'].rolling(window=9).min()
        high9w = df_w['High'].rolling(window=9).max()
        rsv_w = (df_w['Close'] - low9w) / (high9w - low9w) * 100
        df_w['WK'] = rsv_w.ewm(com=2, adjust=False).mean()
        df = df.join(df_w[['WK']], how='left').ffill()
        
        for m in [60, 120]: df[f'MA{m}'] = df['Close'].rolling(window=m).mean()
        return df

    def evaluate_strategy_today(self, ticker, df, war_chest, portfolio_status):
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
        
        month_base_invested = portfolio_status.get("month_base_invested", 0)
        base_done = month_base_invested >= base_alloc * 0.9
        if not base_done:
            sched = self.xtai.schedule(start_date=today, end_date=today + timedelta(days=10))
            is_last_day = (sched.index[0].month != sched.index[1].month) if len(sched) > 1 else True
            dc = False
            for lb in range(1, 4):
                if idx-lb<0: continue
                if df.iloc[idx-lb]['DIF'] > df.iloc[idx-lb]['DEA'] and df.iloc[idx-lb+1]['DIF'] < df.iloc[idx-lb+1]['DEA']:
                    dc = True; break
            if (dc and last['Close'] < prev['Close'] and last['OSC'] < 0) or is_last_day:
                suggestion = "建議基礎投資"
                invest_amt += (base_alloc - month_base_invested)

        executed_extra = portfolio_status.get("executed_extra", [])
        avg_cost = portfolio_status.get("avg_cost", 0)
        mode = t_conf["mode"]
        
        extra_amt = 0
        if mode == "PYRAMID" and avg_cost > 0:
            drop = (last['Close'] - avg_cost) / avg_cost
            for s_name, s_cfg in self.config["pyramid_levels"].items():
                if drop <= s_cfg["drop"] and s_name not in executed_extra:
                    req_amt = base_alloc * s_cfg["mult"]
                    if war_chest >= req_amt:
                        extra_amt = req_amt
                        suggestion = f"建議金字塔加碼({s_name})" if suggestion=="觀望" else suggestion + f" & 加碼({s_name})"
                        break
        elif mode == "TECH":
            triggered = False
            if (last['K'] < 20 or last['WK'] < 20) and "K_OVER" not in executed_extra:
                if war_chest >= base_alloc:
                    extra_amt += base_alloc
                    suggestion = "建議K值加碼" if suggestion=="觀望" else suggestion + " & K值加碼"
                    triggered = True
            if not triggered:
                for ma in ['MA60', 'MA120']:
                    mv = last[ma]
                    if last['Close'] >= mv and (last['Close']-mv)/mv < 0.02 and (last['Low'] <= mv or prev['Low'] <= mv):
                        if ma not in executed_extra:
                            if war_chest >= base_alloc:
                                extra_amt += base_alloc
                                suggestion = f"建議{ma}加碼" if suggestion=="觀望" else suggestion + f" & {ma}加碼"
                                break
        return suggestion, invest_amt + extra_amt

    def run_backtest(self):
        start_date = self.config["backtest_start_date"]
        print(f">>> 啟動回測引擎 | 起始日: {start_date}")
        tickers = list(self.config["targets"].keys())
        data_map = {}
        for t in tickers:
            raw = yf.download(t, start=pd.to_datetime(start_date) - timedelta(days=200))
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
                row = df.loc[date]
                idx = df.index.get_loc(date)
                price = row['Close']
                prev = df.iloc[idx-1]
                t_conf = self.config["targets"][t]
                base_budget = budget * t_conf["ratio"]

                if not month_base_done[t]:
                    is_last = (date.month != (date + timedelta(days=5)).month)
                    dc = False
                    for lb in range(1, 4):
                        if idx-lb<0: continue
                        if df.iloc[idx-lb]['DIF'] > df.iloc[idx-lb]['DEA'] and df.iloc[idx-lb+1]['DIF'] < df.iloc[idx-lb+1]['DEA']:
                            dc = True; break
                    if (dc and price < prev['Close'] and row['OSC'] < 0) or is_last:
                        sh = base_budget / price
                        portfolio[t]["shares"] += sh
                        portfolio[t]["cost"] += base_budget
                        month_base_done[t] = True
                        history.append({"日期": date.strftime('%Y-%m-%d'), "標的": t, "策略": "基礎投資", "金額": int(base_budget), "股數": round(sh, 2)})

                avg = portfolio[t]["cost"] / portfolio[t]["shares"] if portfolio[t]["shares"] > 0 else 0
                mode = t_conf["mode"]
                if mode == "PYRAMID" and avg > 0:
                    drop = (price - avg) / avg
                    for s_name, s_cfg in self.config["pyramid_levels"].items():
                        if drop <= s_cfg["drop"] and s_name not in extra_done[t]:
                            amt = base_budget * s_cfg["mult"]
                            if war_chest >= amt:
                                sh = amt / price
                                portfolio[t]["shares"] += sh
                                portfolio[t]["cost"] += amt
                                war_chest -= amt
                                extra_done[t].add(s_name)
                                history.append({"日期": date.strftime('%Y-%m-%d'), "標的": t, "策略": f"金字塔_{s_name}", "金額": int(amt), "股數": round(sh, 2)})
                                break
                elif mode == "TECH":
                    if (row['K'] < 20 or row['WK'] < 20) and "K_OVER" not in extra_done[t]:
                        if war_chest >= base_budget:
                            sh = base_budget / price
                            portfolio[t]["shares"] += sh
                            portfolio[t]["cost"] += base_budget
                            war_chest -= base_budget
                            extra_done[t].add("K_OVER")
                    for ma in ['MA60', 'MA120']:
                        mv = row[ma]
                        if price >= mv and (price-mv)/mv < 0.02 and (row['Low'] <= mv or df.iloc[idx-1]['Low'] <= mv):
                            if ma not in extra_done[t]:
                                if war_chest >= base_budget:
                                    sh = base_budget / price
                                    portfolio[t]["shares"] += sh
                                    portfolio[t]["cost"] += base_budget
                                    war_chest -= base_budget
                                    extra_done[t].add(ma)
                                    history.append({"日期": date.strftime('%Y-%m-%d'), "標的": t, "策略": f"{ma}有撐", "金額": int(base_budget), "股數": round(sh, 2)})

        res_df = pd.DataFrame(history)
        return res_df, war_chest

    def generate_chart(self, summary_df, cash_tickers):
        try:
            plot_df = summary_df[~summary_df['標的'].isin(cash_tickers)].copy()
            if plot_df.empty: return None

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            x = np.arange(len(plot_df))
            width = 0.35
            rects1 = axes[0].bar(x - width/2, plot_df['金額'], width, label='投入成本', color='#95a5a6')
            rects2 = axes[0].bar(x + width/2, plot_df['MarketValue'], width, label='目前市值', color='#e74c3c')
            
            axes[0].set_ylabel('金額 (TWD)')
            axes[0].set_title('投資組合: 成本 vs 市值')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(plot_df['標的'], rotation=45)
            axes[0].legend()
            axes[0].grid(axis='y', linestyle='--', alpha=0.7)

            axes[1].pie(plot_df['MarketValue'], labels=plot_df['標的'], autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
            axes[1].set_title('資產市值配置比例')

            plt.tight_layout()
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', dpi=100)
            img_buf.seek(0)
            plt.close()
            return img_buf
        except:
            return None

    def analyze_and_notify(self, df, mode="REAL", war_chest_sim=0):
        print(f">>> 執行資產分析與策略報告生成 [{mode}]...")
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
        
        for idx, row in summary.iterrows():
            ticker = row['標的']
            if ticker in cash_tickers or ticker.upper() == 'CASH':
                if mode == "REAL" and 'RowMV' in summary.columns:
                    summary.at[idx, 'MarketValue'] = row['RowMV']
                    summary.at[idx, 'Price'] = (row['RowMV'] / row['股數']) if row['股數'] > 0 else 1.0
                else:
                    summary.at[idx, 'MarketValue'] = war_chest_sim
                    summary.at[idx, 'Price'] = 1.0
            else:
                try:
                    curr_data = yf.download(ticker, period="1d", progress=False)
                    if isinstance(curr_data.columns, pd.MultiIndex):
                        curr_data.columns = curr_data.columns.get_level_values(0)
                    price = float(curr_data['Close'].iloc[-1]) if not curr_data.empty else 0
                except: price = 0
                summary.at[idx, 'Price'] = price
                summary.at[idx, 'MarketValue'] = row['股數'] * price

        chart_bytes = self.generate_chart(summary, cash_tickers)

        html = f"<h2>智投報告 [{mode}] - {now.strftime('%Y-%m-%d')}</h2>"
        html += '<img src="cid:portfolio_chart" alt="Portfolio Chart" style="max-width:100%;"><br><hr>'
        html += "<table border='1' cellpadding='5' style='border-collapse:collapse; width:100%; font-family: Arial;'>"
        html += "<tr style='background:#f2f2f2;'><th>標的</th><th>股數</th><th>市值</th><th>報酬率</th>"
        if mode == "REAL":
            html += "<th>今日建議</th><th>建議金額</th><th>相當股數</th>"
        html += "</tr>"

        print(f"\n{'標的名稱':<12} {'股數':>8} {'市值':>10} {'報酬率':>8} {'今日建議':<10}")
        all_targets = sorted(list(set(list(summary['標的']) + list(self.config["targets"].keys()))))
        
        total_cost = 0
        total_mv = 0

        for ticker in all_targets:
            if ticker == 'CASH': continue
            row = summary[summary['標的'] == ticker]
            if not row.empty:
                cost = row.iloc[0]['金額']
                shares = row.iloc[0]['股數']
                mv = row.iloc[0]['MarketValue']
            else:
                cost=0; shares=0; mv=0
            
            roi = (mv - cost) / cost * 100 if cost > 0 else 0
            total_cost += cost
            total_mv += mv
            
            is_fd = ticker in cash_tickers
            suggestion = "-"; sugg_amt_str = "-"; sugg_shares_str = "-"
            
            if mode == "REAL" and not is_fd:
                avg_cost = cost / shares if shares > 0 else 0
                df['日期'] = pd.to_datetime(df['日期'])
                month_df = df[(df['標的'] == ticker) & (df['日期'].dt.month == now.month) & (df['日期'].dt.year == now.year)]
                col_type = next((c for c in ['策略', '類別', '類型'] if c in df.columns), None)
                
                month_base = 0; extra_types = []
                if col_type:
                    month_base = month_df[month_df[col_type].astype(str).str.contains('基礎', na=False)]['金額'].sum()
                    for t_type in month_df[col_type].unique():
                        t_str = str(t_type)
                        if "金字塔" in t_str: extra_types.append(t_str.split('_')[-1])
                        if "技術" in t_str or "K值" in t_str: extra_types.append("K_OVER" if "K值" in t_str else t_str)
                
                try:
                    hist_data = yf.download(ticker, period="2y", progress=False)
                    if isinstance(hist_data.columns, pd.MultiIndex): hist_data.columns = hist_data.columns.get_level_values(0)
                    if not hist_data.empty:
                        curr_price = float(hist_data['Close'].iloc[-1])
                        hist_data = self.calculate_indicators(hist_data)
                        p_status = {"month_base_invested": month_base, "executed_extra": extra_types, "avg_cost": avg_cost}
                        s_text, s_val = self.evaluate_strategy_today(ticker, hist_data, war_chest_mv, p_status)
                        suggestion = s_text
                        if s_val > 0:
                            sugg_amt_str = f"{s_val:,.0f}"
                            sugg_shares_str = f"{int(s_val / curr_price)}"
                except: pass

            t_name = self.config["targets"].get(ticker, {}).get("name", ticker)
            print(f"{t_name:<12} {shares:>8.0f} {mv:>10,.0f} {roi:>7.2f}% {suggestion:<10}")
            
            roi_color = "red" if roi > 0 else "green"
            html += f"<tr><td>{t_name}</td><td>{shares:,.0f}</td><td>{mv:,.0f}</td><td style='color:{roi_color}'>{roi:+.2f}%</td>"
            if mode == "REAL":
                if is_fd:
                    html += "<td align='center'>-</td><td align='center'>-</td><td align='center'>-</td>"
                else:
                    s_color = "red" if sugg_amt_str != "-" else "black"
                    html += f"<td style='color:{s_color};'>{suggestion}</td><td style='color:{s_color};'>{sugg_amt_str}</td><td>{sugg_shares_str}</td>"
            html += "</tr>"

        tot_roi = (total_mv - total_cost) / total_cost * 100 if total_cost > 0 else 0
        html += "</table>"
        html += f"<br><b>總投入成本:</b> {total_cost:,.0f} TWD<br>"
        html += f"<b>總資產市值:</b> {total_mv:,.0f} TWD<br>"
        wc_display = war_chest_mv if mode == "REAL" else war_chest_sim
        html += f"<b>加碼金餘額:</b> <span style='color:blue'>{wc_display:,.0f} TWD</span><br>"
        html += f"<b>整體報酬率:</b> <span style='font-size:18px; color:{'red' if tot_roi>0 else 'green'}'>{tot_roi:+.2f}%</span><br>"
        
        email_conf = self.config.get("email_config", {})
        if email_conf.get("enable") and email_conf.get("receiver_email"):
            subject = f"智投報告 [{mode}] - {now.strftime('%Y-%m-%d')}"
            self.rm.send_email_with_chart(email_conf["receiver_email"], subject, html, chart_bytes)

    def run(self):
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