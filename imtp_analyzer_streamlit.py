import streamlit as st
import pandas as pd
import numpy as np
from scipy import signal
import plotly.graph_objects as go
import io
import csv
import base64
from datetime import datetime
import time

# ページ設定 - ワイドモードで表示
st.set_page_config(
    page_title="IMTP分析アプリケーション",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSSでスタイル設定
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-text {
        color: #3498db;
        font-weight: bold;
    }
    .warning-text {
        color: #e74c3c;
        font-weight: bold;
    }
    .success-text {
        color: #2ecc71;
        font-weight: bold;
    }
    .black-text {
        color: #000000;
    }
    .block-container {
        padding-top: 1rem;
    }
    .stDataFrame {
        font-size: 0.8rem;
    }
    div[data-testid="column"] {
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        padding: 10px;
        border-radius: 5px;
        background-color: #f8f9fa;
    }
    div[data-testid="stVerticalBlock"] > div:has(div.element-container div.stDataFrame) {
        overflow-x: scroll;
    }
    .onset-adjustment {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# アプリケーションタイトル
st.markdown('<h1 class="main-header">IMTP 分析ツール</h1>', unsafe_allow_html=True)

# サイドバーに設定パラメータを配置
st.sidebar.header("パラメータ設定")
onset_threshold = st.sidebar.slider("Onset閾値(SD×):", 1.0, 10.0, 5.0, 0.1, help="Onset検出のための標準偏差倍率")
filter_freq = st.sidebar.slider("フィルター(Hz):", 10.0, 100.0, 50.0, 1.0, help="バターワースローパスフィルターのカットオフ周波数")
sampling_rate = st.sidebar.number_input("サンプリングレート (Hz):", min_value=100, max_value=10000, value=1000, step=100, help="データのサンプリングレート")
baseline_window = int(sampling_rate)  # サンプリングレートに基づくベースラインウィンドウ設定（1秒分）

# セッション状態の初期化
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'selected_trial' not in st.session_state:
    st.session_state['selected_trial'] = 0
if 'trial_data' not in st.session_state:
    st.session_state['trial_data'] = []
if 'trial_names' not in st.session_state:
    st.session_state['trial_names'] = []
if 'trial_results' not in st.session_state:
    st.session_state['trial_results'] = []
if 'manual_onset_adjustments' not in st.session_state:
    st.session_state['manual_onset_adjustments'] = {}
if 'auto_onset_times' not in st.session_state:
    st.session_state['auto_onset_times'] = {}
if 'apply_onset_change' not in st.session_state:
    st.session_state['apply_onset_change'] = False
if 'force_show_results' not in st.session_state:
    st.session_state['force_show_results'] = False
if 'analysis_completed' not in st.session_state:
    st.session_state['analysis_completed'] = False
if 'current_view' not in st.session_state:
    st.session_state['current_view'] = 'input'  # 'input' または 'results'

# 解析関数
def apply_filter(force_data, filter_freq, sampling_rate, filter_order=4):
    """バターワースローパスフィルターを適用する関数"""
    nyquist = 0.5 * sampling_rate
    cutoff = filter_freq / nyquist
    b, a = signal.butter(filter_order, cutoff, btype='low')
    filtered_data = signal.filtfilt(b, a, force_data)
    return filtered_data

def detect_onset(force_data, baseline_window, onset_threshold):
    """力発揮開始点（Onset）を検出する関数"""
    # ベースラインの計算（最初の1秒間のデータを使用）
    baseline_data = force_data[:baseline_window]
    baseline_mean = np.mean(baseline_data)
    baseline_std = np.std(baseline_data)
    
    # Onset閾値の計算
    threshold = baseline_mean + (baseline_std * onset_threshold)
    
    # 閾値を超えた最初のインデックスを検索
    for i in range(len(force_data)):
        if force_data[i] > threshold:
            # 連続して5点が閾値を超えているか確認（ノイズ対策）
            if i + 5 < len(force_data) and all(f > threshold for f in force_data[i:i+5]):
                return i, baseline_mean, threshold
    
    # 見つからない場合は0を返す
    return 0, baseline_mean, threshold

def calculate_rfd(force_data, onset_index, sampling_rate):
    """指定された時間枠でのRFD（Rate of Force Development）を計算する関数"""
    rfd_results = {}
    
    # 各時間枠でのRFDを計算
    time_windows = [50, 100, 150, 200, 250]  # ミリ秒単位
    
    # onset時の力の値
    force_at_onset = force_data[onset_index]
    
    for window in time_windows:
        # 時間枠に対応するデータポイント数を計算（サンプリング周波数に基づく）
        points = int(window * sampling_rate / 1000)
        
        # onset_indexからpoints分のデータがあるか確認
        if onset_index + points < len(force_data):
            # 各時間点での力
            force_at_timepoint = force_data[onset_index + points]
            
            # 力の変化（各時間点の力 - onset時の力）
            force_change = force_at_timepoint - force_at_onset
            
            # RFD計算（N/s）- 力の変化を時間（秒単位）で割る
            rfd = force_change / (window / 1000)
            rfd_results[f"RFD 0-{window}ms"] = rfd
        else:
            rfd_results[f"RFD 0-{window}ms"] = None
            
    return rfd_results

def time_to_index(time_seconds, sampling_rate):
    """時間（秒）をインデックスに変換"""
    return int(time_seconds * sampling_rate)

def index_to_time(index, sampling_rate):
    """インデックスを時間（秒）に変換"""
    return index / sampling_rate

def analyze_trial(time_data, force_data, filter_freq, onset_threshold, sampling_rate, baseline_window, manual_onset_time=None):
    """試技を分析する関数（手動Onset調整対応）"""
    # フィルターを適用
    filtered_force = apply_filter(force_data, filter_freq, sampling_rate)
    
    # 自動Onset検出
    auto_onset_index, baseline_mean, threshold = detect_onset(filtered_force, baseline_window, onset_threshold)
    auto_onset_time = index_to_time(auto_onset_index, sampling_rate)
    
    # 手動調整が指定されている場合はそれを使用、そうでなければ自動検出を使用
    if manual_onset_time is not None:
        onset_index = time_to_index(manual_onset_time, sampling_rate)
        onset_time = manual_onset_time
    else:
        onset_index = auto_onset_index
        onset_time = auto_onset_time
    
    # インデックスの範囲チェック
    onset_index = max(0, min(onset_index, len(filtered_force) - 1))
    
    # ピーク力を取得（Onset以降のデータから）
    if onset_index < len(filtered_force):
        peak_force = np.max(filtered_force[onset_index:])
        peak_force_index = np.argmax(filtered_force[onset_index:]) + onset_index
    else:
        peak_force = np.max(filtered_force)
        peak_force_index = np.argmax(filtered_force)
    
    # Onset時の力
    onset_force = filtered_force[onset_index]
    
    # Time to Peak（Onsetからピークまでの時間）
    time_to_peak = (peak_force_index - onset_index) / sampling_rate
    
    # RFDを計算
    rfd_values = calculate_rfd(filtered_force, onset_index, sampling_rate)
    
    # 結果を辞書として返す
    return {
        'baseline_mean': baseline_mean,
        'threshold': threshold,
        'auto_onset_index': auto_onset_index,
        'auto_onset_time': auto_onset_time,
        'onset_index': onset_index,
        'onset_time': onset_time,
        'onset_force': onset_force,
        'peak_force': peak_force,
        'peak_force_index': peak_force_index,
        'peak_time': peak_force_index / sampling_rate,
        'net_peak_force': peak_force - baseline_mean,
        'time_to_peak': time_to_peak,
        'rfd_values': rfd_values,
        'filtered_force': filtered_force.tolist(),  # json serializable
        'time_data': time_data.tolist(),  # json serializable
        'manual_adjustment': manual_onset_time is not None
    }

# ファイルアップロードセクション
st.markdown('<h2 class="sub-header">データ入力</h2>', unsafe_allow_html=True)

upload_col1, upload_col2 = st.columns(2)

with upload_col1:
    # 単一試技ファイルアップロード
    uploaded_file = st.file_uploader("単一試技ファイルをアップロード", type=["csv", "txt", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # ファイル拡張子によって読み込み方法を変更
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                try:
                    data = pd.read_csv(uploaded_file)
                except:
                    data = pd.read_csv(uploaded_file, sep=None, engine='python')
            elif file_extension == 'txt':
                try:
                    data = pd.read_csv(uploaded_file, delimiter='\t')
                except:
                    try:
                        data = pd.read_csv(uploaded_file, delimiter=',')
                    except:
                        data = pd.read_csv(uploaded_file, delimiter=' ', skipinitialspace=True)
            elif file_extension in ['xlsx', 'xls']:
                data = pd.read_excel(uploaded_file)
            
            # データ前処理
            # 空白や不正な文字を含む列名の修正
            data.columns = [str(col).strip().replace(' ', '_').replace('\n', '') for col in data.columns]
            
            # データ型変換（文字列を数値に）
            for col in data.columns:
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                except:
                    pass
            
            # 欠損値を処理
            data = data.fillna(method='ffill')
            
            # セッション状態にデータを保存
            st.session_state['data'] = data
            st.session_state['trial_data'] = [data]
            st.session_state['trial_names'] = ["単一試技"]
            st.session_state['trial_results'] = [None]
            st.session_state['selected_trial'] = 0
            
            st.success(f"ファイルを読み込みました: {uploaded_file.name}")
        
        except Exception as e:
            st.error(f"ファイル読み込み中にエラーが発生しました: {str(e)}")

with upload_col2:
    # 複数試技ファイルアップロード
    uploaded_multi_file = st.file_uploader("複数試技ファイルをアップロード", type=["csv", "xlsx"])
    
    if uploaded_multi_file is not None:
        try:
            # ファイル拡張子によって読み込み方法を変更
            file_extension = uploaded_multi_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                raw_data = pd.read_csv(uploaded_multi_file)
            elif file_extension in ['xlsx', 'xls']:
                raw_data = pd.read_excel(uploaded_multi_file)
            
            # 列名の空白や不正な文字を修正
            raw_data.columns = [str(col).strip().replace(' ', '_').replace('\n', '') for col in raw_data.columns]
            
            # データ前処理
            for col in raw_data.columns:
                try:
                    raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce')
                except:
                    pass
            
            # 欠損値処理
            raw_data = raw_data.fillna(method='ffill')
            
            # 列名から時間列を特定 (通常は1列目が時間)
            time_column = raw_data.columns[0]
            time_data = raw_data[time_column].values
            
            # 残りの列は全て力データ (試技) として扱う
            trial_data = []
            trial_names = []
            
            for col in raw_data.columns[1:]:  # 最初の列以外をすべて試技データとして処理
                # 列名を試技名として保存
                trial_names.append(col)
                
                # 時間と力のデータセットを作成
                trial_df = pd.DataFrame({
                    'time': time_data,
                    'force': raw_data[col].values
                })
                
                # 試技データをリストに追加
                trial_data.append(trial_df)
            
            if trial_data:
                # セッション状態にデータを保存
                st.session_state['trial_data'] = trial_data
                st.session_state['trial_names'] = trial_names
                st.session_state['trial_results'] = [None] * len(trial_data)
                st.session_state['selected_trial'] = 0
                st.session_state['data'] = trial_data[0]  # 最初の試技をデフォルトとして選択
                
                st.success(f"{len(trial_data)}件の試技データを読み込みました")
            else:
                st.error("有効な試技データが見つかりませんでした。ファイル形式を確認してください。")
                
        except Exception as e:
            st.error(f"複数試技データ読み込み中にエラーが発生しました: {str(e)}")

# テキストデータ入力オプション
text_expander = st.expander("テキストデータを直接貼り付け")
with text_expander:
    text_data = st.text_area("データをここに貼り付けてください（タブまたはカンマ区切り）:", height=150)
    
    text_col1, text_col2, text_col3 = st.columns(3)
    
    with text_col1:
        delimiter_options = {
            "自動検出": "auto",
            "タブ": "\t",
            "カンマ": ",",
            "スペース": " "
        }
        delimiter = st.selectbox("区切り文字:", list(delimiter_options.keys()))
    
    with text_col2:
        has_header = st.checkbox("1行目をヘッダーとして使用", value=True)
    
    with text_col3:
        convert_numeric = st.checkbox("文字列を数値に変換", value=True)
    
    if st.button("テキストデータを読み込む"):
        if text_data.strip():
            try:
                # エクセルからコピーした際の前処理
                # 改行文字の正規化
                text_data = text_data.replace('\r\n', '\n').replace('\r', '\n')
                
                # 全角記号を半角に変換
                text_data = text_data.replace('．', '.').replace('，', ',').replace('　', ' ')
                
                # 区切り文字を決定
                del_value = delimiter_options[delimiter]
                if del_value == "auto":
                    if "\t" in text_data:
                        del_value = "\t"
                    elif "," in text_data:
                        del_value = ","
                    else:
                        del_value = " "
                
                # ヘッダー設定
                header_value = 0 if has_header else None
                
                # データ読み込み
                data = pd.read_csv(io.StringIO(text_data), sep=del_value, header=header_value, engine='python', on_bad_lines='skip')
                
                # ヘッダーがない場合は列名を自動生成
                if header_value is None:
                    data.columns = [f"Column_{i}" for i in range(len(data.columns))]
                
                # 列名の修正
                data.columns = [str(col).strip().replace(' ', '_').replace('\n', '') for col in data.columns]
                
                # データ型変換
                if convert_numeric:
                    for col in data.columns:
                        try:
                            data[col] = pd.to_numeric(data[col], errors='coerce')
                        except:
                            pass
                
                # 欠損値処理
                data = data.fillna(method='ffill')
                
                # セッション状態にデータを保存
                st.session_state['data'] = data
                st.session_state['trial_data'] = [data]
                st.session_state['trial_names'] = ["テキストデータ"]
                st.session_state['trial_results'] = [None]
                st.session_state['selected_trial'] = 0
                
                st.success(f"テキストデータを読み込みました。行数: {data.shape[0]}, 列数: {data.shape[1]}")
                
            except Exception as e:
                st.error(f"テキストデータ読み込み中にエラーが発生しました: {str(e)}")
        else:
            st.warning("テキストが入力されていません。")

# データが読み込まれている場合、現在のビューに応じて表示を切り替え
if st.session_state['data'] is not None:
    
    # ビュー切り替えボタンを最上部に配置
    view_col1, view_col2, view_col3 = st.columns([1, 1, 2])
    
    with view_col1:
        if st.button("📊 データ入力・分析", key="switch_to_input", use_container_width=True):
            st.session_state['current_view'] = 'input'
            st.rerun()
    
    with view_col2:
        # 結果があるかチェック
        has_results = False
        try:
            if (st.session_state['trial_results'] and 
                st.session_state['selected_trial'] < len(st.session_state['trial_results']) and
                st.session_state['trial_results'][st.session_state['selected_trial']] is not None):
                has_results = True
        except (IndexError, TypeError):
            has_results = False
        
        if has_results:
            if st.button("📈 分析結果・調整", key="switch_to_results", use_container_width=True):
                st.session_state['current_view'] = 'results'
                st.rerun()
        else:
            st.button("📈 分析結果・調整", key="switch_to_results_disabled", disabled=True, use_container_width=True)
    
    with view_col3:
        # 現在のビューを表示
        if st.session_state['current_view'] == 'input':
            st.markdown("**現在のビュー:** データ入力・分析")
        else:
            st.markdown("**現在のビュー:** 分析結果・調整")
    
    st.markdown("---")
    
    # ビューに応じた表示を開始
    if st.session_state['current_view'] == 'input':
        # === データ入力・分析ビュー ===
        
        # データプレビュー
        st.markdown('<h2 class="sub-header">データプレビュー</h2>', unsafe_allow_html=True)
        st.dataframe(st.session_state['data'].head(5), use_container_width=True)
        
        # 試技選択（複数試技がある場合）
        if len(st.session_state['trial_names']) > 1:
            st.markdown('<h2 class="sub-header">試技選択</h2>', unsafe_allow_html=True)
            
            trial_col1, trial_col2 = st.columns([3, 1])
            
            with trial_col1:
                selected_trial_name = st.selectbox(
                    "試技:", 
                    st.session_state['trial_names'],
                    index=st.session_state['selected_trial']
                )
                selected_trial_index = st.session_state['trial_names'].index(selected_trial_name)
                st.session_state['selected_trial'] = selected_trial_index
                st.session_state['data'] = st.session_state['trial_data'][selected_trial_index]
            
            with trial_col2:
                trial_nav_col1, trial_nav_col2 = st.columns(2)
                
                with trial_nav_col1:
                    if st.button("◀ 前へ", key="prev_trial"):
                        prev_idx = (st.session_state['selected_trial'] - 1) % len(st.session_state['trial_data'])
                        st.session_state['selected_trial'] = prev_idx
                        st.session_state['data'] = st.session_state['trial_data'][prev_idx]
                        st.rerun()
                
                with trial_nav_col2:
                    if st.button("次へ ▶", key="next_trial"):
                        next_idx = (st.session_state['selected_trial'] + 1) % len(st.session_state['trial_data'])
                        st.session_state['selected_trial'] = next_idx
                        st.session_state['data'] = st.session_state['trial_data'][next_idx]
                        st.rerun()
        
        # 分析パラメータ設定
        st.markdown('<h2 class="sub-header">列選択</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 時間列選択
            time_column = st.selectbox(
                "時間列:",
                st.session_state['data'].columns.tolist(),
                index=0
            )
        
        with col2:
            # 力列選択
            force_column = st.selectbox(
                "力列:",
                st.session_state['data'].columns.tolist(),
                index=min(1, len(st.session_state['data'].columns)-1)
            )
        
        # 分析実行ボタン
        st.markdown('<h2 class="sub-header">分析実行</h2>', unsafe_allow_html=True)
        
        button_col1, button_col2 = st.columns(2)
        
        with button_col1:
            if st.button("現在の試技を分析", type="primary", use_container_width=True):
                try:
                    # データ取得
                    try:
                        time_data = st.session_state['data'][time_column].values
                        force_data = st.session_state['data'][force_column].values
                    except KeyError as e:
                        st.error(f"選択された列 '{time_column}' または '{force_column}' がデータに見つかりません。")
                        st.stop()
                        
                    with st.spinner('分析中...'):
                        # 試技のキーを作成
                        trial_key = f"{st.session_state['selected_trial']}_{st.session_state['trial_names'][st.session_state['selected_trial']]}"
                        
                        # 手動調整があるかチェック
                        manual_onset = st.session_state['manual_onset_adjustments'].get(trial_key, None)
                        
                        # 分析実行
                        result = analyze_trial(
                            time_data, 
                            force_data, 
                            filter_freq, 
                            onset_threshold, 
                            sampling_rate, 
                            baseline_window,
                            manual_onset_time=manual_onset
                        )
                        
                        # 自動検出されたOnset時間を保存
                        st.session_state['auto_onset_times'][trial_key] = result['auto_onset_time']
                        
                        # 結果を保存
                        st.session_state['trial_results'][st.session_state['selected_trial']] = result
                        
                        # 分析完了フラグを設定
                        st.session_state['analysis_completed'] = True
                        st.session_state['force_show_results'] = True
                        st.session_state['current_view'] = 'results'  # 結果ビューに自動切り替え
                        
                        # 成功メッセージを表示
                        st.success('分析が完了しました！結果ビューに切り替わります。')
                        st.rerun()
                
                except Exception as e:
                    st.error(f"分析中にエラーが発生しました: {str(e)}")
                    # エラーの詳細情報を表示
                    import traceback
                    st.code(traceback.format_exc(), language="python")
        
        with button_col2:
            if len(st.session_state['trial_data']) > 1:
                analyze_all = st.button("全試技を分析", use_container_width=True)
                
                if analyze_all:
                    try:
                        # 進捗バーの設定
                        progress_container = st.container()
                        
                        with progress_container:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # 分析結果を一時的に保存するリスト
                            temp_results = []
                            
                            # 全試技の分析
                            for i, trial_df in enumerate(st.session_state['trial_data']):
                                # 進捗表示更新
                                progress = (i) / len(st.session_state['trial_data'])
                                progress_bar.progress(progress)
                                status_text.text(f"試技 {i+1}/{len(st.session_state['trial_data'])} を分析中: {st.session_state['trial_names'][i]}")
                                
                                # データ取得と分析
                                try:
                                    # 必要な列の存在を確認
                                    if 'time' in trial_df.columns and 'force' in trial_df.columns:
                                        time_data = trial_df['time'].values
                                        force_data = trial_df['force'].values
                                    else:
                                        # 最初の列を時間、2番目の列を力と想定
                                        cols = trial_df.columns.tolist()
                                        if len(cols) >= 2:
                                            time_data = trial_df[cols[0]].values
                                            force_data = trial_df[cols[1]].values
                                        else:
                                            raise ValueError(f"試技 {i+1} には十分な列がありません")
                                    
                                    # 試技のキーを作成
                                    trial_key = f"{i}_{st.session_state['trial_names'][i]}"
                                    
                                    # 手動調整があるかチェック
                                    manual_onset = st.session_state['manual_onset_adjustments'].get(trial_key, None)
                                    
                                    # 分析実行
                                    result = analyze_trial(
                                        time_data, 
                                        force_data, 
                                        filter_freq, 
                                        onset_threshold, 
                                        sampling_rate, 
                                        baseline_window,
                                        manual_onset_time=manual_onset
                                    )
                                    
                                    # 自動検出されたOnset時間を保存
                                    st.session_state['auto_onset_times'][trial_key] = result['auto_onset_time']
                                    
                                    # 結果を一時リストに追加
                                    temp_results.append(result)
                                    
                                except Exception as e:
                                    st.error(f"試技 {i+1} の分析中にエラーが発生しました: {str(e)}")
                                    temp_results.append(None)  # エラーの場合はNoneを追加
                            
                            # 分析が完了したらセッション状態に一括で保存
                            if len(temp_results) == len(st.session_state['trial_data']):
                                st.session_state['trial_results'] = temp_results
                            
                            # 完了表示
                            progress_bar.progress(1.0)
                            status_text.text(f"{len(st.session_state['trial_data'])}件の試技データを分析完了")
                            
                            # 分析完了メッセージを表示
                            st.success('全試技の分析が完了しました！結果ビューに切り替えます。')
                            st.session_state['current_view'] = 'results'  # 結果ビューに自動切り替え
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"全試技分析中にエラーが発生しました: {str(e)}")
                        # エラーの詳細情報を表示
                        import traceback
                        st.code(traceback.format_exc(), language="python")

    elif st.session_state['current_view'] == 'results':
        # === 分析結果・調整ビュー ===
        
        # 結果表示条件を確認
        current_result = None
        should_show_results = False
        
        if st.session_state['trial_results'] and st.session_state['selected_trial'] < len(st.session_state['trial_results']):
            current_result = st.session_state['trial_results'][st.session_state['selected_trial']]
            if current_result is not None:
                should_show_results = True
        
        if not should_show_results:
            st.warning("表示する分析結果がありません。先にデータ入力・分析ビューで分析を実行してください。")
            if st.button("データ入力・分析ビューに戻る"):
                st.session_state['current_view'] = 'input'
                st.rerun()
        else:
            # 試技選択（複数試技がある場合）
            if len(st.session_state['trial_names']) > 1:
                st.markdown('<h2 class="sub-header">試技選択</h2>', unsafe_allow_html=True)
                
                trial_col1, trial_col2 = st.columns([3, 1])
                
                with trial_col1:
                    selected_trial_name = st.selectbox(
                        "試技:", 
                        st.session_state['trial_names'],
                        index=st.session_state['selected_trial'],
                        key="results_trial_select"
                    )
                    selected_trial_index = st.session_state['trial_names'].index(selected_trial_name)
                    if selected_trial_index != st.session_state['selected_trial']:
                        st.session_state['selected_trial'] = selected_trial_index
                        st.session_state['data'] = st.session_state['trial_data'][selected_trial_index]
                        st.rerun()
                
                with trial_col2:
                    trial_nav_col1, trial_nav_col2 = st.columns(2)
                    
                    with trial_nav_col1:
                        if st.button("◀ 前へ", key="prev_trial_results"):
                            prev_idx = (st.session_state['selected_trial'] - 1) % len(st.session_state['trial_data'])
                            st.session_state['selected_trial'] = prev_idx
                            st.session_state['data'] = st.session_state['trial_data'][prev_idx]
                            st.rerun()
                    
                    with trial_nav_col2:
                        if st.button("次へ ▶", key="next_trial_results"):
                            next_idx = (st.session_state['selected_trial'] + 1) % len(st.session_state['trial_data'])
                            st.session_state['selected_trial'] = next_idx
                            st.session_state['data'] = st.session_state['trial_data'][next_idx]
                            st.rerun()
            
            # 現在の結果を取得
            current_result = st.session_state['trial_results'][st.session_state['selected_trial']]
            
            st.markdown('<h2 class="sub-header">分析結果</h2>', unsafe_allow_html=True)
            
            # 列選択情報を取得（分析時の設定を使用）
            # デフォルト値を設定
            time_column = st.session_state['data'].columns[0] if len(st.session_state['data'].columns) > 0 else 'time'
            force_column = st.session_state['data'].columns[1] if len(st.session_state['data'].columns) > 1 else 'force'
            
            # Onset手動調整セクション（簡素化版）
            st.markdown('<div class="onset-adjustment">', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">🎯 Onset タイミング調整</h3>', unsafe_allow_html=True)
            
            # 現在の試技のキー
            trial_key = f"{st.session_state['selected_trial']}_{st.session_state['trial_names'][st.session_state['selected_trial']]}"
            
            onset_col1, onset_col2, onset_col3 = st.columns([2, 2, 1])
            
            with onset_col1:
                # 自動検出されたOnset時間を表示
                auto_onset_time = current_result['auto_onset_time']
                st.markdown(f"**自動検出Onset:** {auto_onset_time:.3f} 秒")
                
                # 現在使用されているOnset時間
                current_onset_time = current_result['onset_time']
                is_manual = current_result.get('manual_adjustment', False)
                
                if is_manual:
                    st.markdown(f"**現在のOnset:** <span class='warning-text'>{current_onset_time:.3f} 秒 (手動調整済み)</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**現在のOnset:** <span class='success-text'>{current_onset_time:.3f} 秒 (自動検出)</span>", unsafe_allow_html=True)
            
            with onset_col2:
                # 調整値の管理（シンプル版）
                adjustment_key = f"temp_onset_{trial_key}"
                if adjustment_key not in st.session_state:
                    st.session_state[adjustment_key] = current_onset_time
                
                # 数値入力（単純な形式）
                new_onset_value = st.number_input(
                    "Onset時間調整 (秒)",
                    value=st.session_state[adjustment_key],
                    step=0.001,
                    format="%.3f",
                    key=f"onset_input_{trial_key}",
                    help="1ms単位で調整可能"
                )
                
                # 値が変更された場合、セッション状態を更新
                if new_onset_value != st.session_state[adjustment_key]:
                    st.session_state[adjustment_key] = new_onset_value
                
                # 差分表示
                diff_ms = (new_onset_value - auto_onset_time) * 1000
                if abs(diff_ms) > 0.5:
                    st.markdown(f"**調整量:** {diff_ms:+.1f} ms")
            
            with onset_col3:
                # 適用ボタン
                if st.button("🔄 適用", key=f"apply_onset_simple_{trial_key}", type="primary"):
                    try:
                        with st.spinner('調整を適用中...'):
                            # 手動調整値を保存
                            st.session_state['manual_onset_adjustments'][trial_key] = new_onset_value
                            
                            # 分析を再実行
                            time_data = st.session_state['data'][time_column].values
                            force_data = st.session_state['data'][force_column].values
                            
                            result = analyze_trial(
                                time_data, 
                                force_data, 
                                filter_freq, 
                                onset_threshold, 
                                sampling_rate, 
                                baseline_window,
                                manual_onset_time=new_onset_value
                            )
                            
                            # 結果を更新
                            st.session_state['trial_results'][st.session_state['selected_trial']] = result
                            st.success("✅ 調整が適用されました！")
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"❌ エラーが発生しました: {str(e)}")
                
                # リセットボタン
                if st.button("↩️ リセット", key=f"reset_onset_simple_{trial_key}"):
                    try:
                        with st.spinner('リセット中...'):
                            # 手動調整を削除
                            if trial_key in st.session_state['manual_onset_adjustments']:
                                del st.session_state['manual_onset_adjustments'][trial_key]
                            
                            # 調整値もリセット
                            st.session_state[adjustment_key] = auto_onset_time
                            
                            # 自動検出で再分析
                            time_data = st.session_state['data'][time_column].values
                            force_data = st.session_state['data'][force_column].values
                            
                            result = analyze_trial(
                                time_data, 
                                force_data, 
                                filter_freq, 
                                onset_threshold, 
                                sampling_rate, 
                                baseline_window,
                                manual_onset_time=None
                            )
                            
                            st.session_state['trial_results'][st.session_state['selected_trial']] = result
                            st.success("✅ 自動検出に戻しました！")
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"❌ エラーが発生しました: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.markdown('<h3 class="sub-header">測定概要</h3>', unsafe_allow_html=True)
                
                # 試技名表示（複数試技モードの場合）
                if len(st.session_state['trial_names']) > 1:
                    st.markdown(f"**試技:** {st.session_state['trial_names'][st.session_state['selected_trial']]}")
                
                st.markdown("**━━━━━━━━━ 基本測定値 ━━━━━━━━━**")
                
                st.markdown(f"**安静時の平均値（体重相当）:** {current_result['baseline_mean']:.2f} N")
                
                st.markdown(f"**Onset時点:** {current_result['onset_time']:.3f} 秒")
                st.markdown(f"**Onset時の力:** {current_result['onset_force']:.2f} N")
                
                st.markdown(f"**Peak Force:** {current_result['peak_force']:.2f} N")
                st.markdown(f"**Net Peak Force（体重差し引き）:** {current_result['net_peak_force']:.2f} N")
                
                st.markdown(f"**Peak Force タイミング:** {current_result['peak_time']:.3f} 秒")
                st.markdown(f"**Onsetからピークまでの時間:** <span class='warning-text'>{current_result['time_to_peak']:.3f} 秒</span>", unsafe_allow_html=True)
                
                st.markdown("**━━━━━━━━━ 分析設定 ━━━━━━━━━**")
                
                st.markdown(f"**フィルター設定:** {filter_freq} Hz, 4次 Butterworth")
                st.markdown(f"**Onset閾値:** ベースライン + {onset_threshold} SD")
                st.markdown(f"**サンプリングレート:** {sampling_rate} Hz")
                
                # 手動調整の状態表示
                if current_result.get('manual_adjustment', False):
                    st.markdown(f"**Onset調整:** <span class='warning-text'>手動調整済み</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Onset調整:** <span class='success-text'>自動検出</span>", unsafe_allow_html=True)
            
            with result_col2:
                st.markdown('<h3 class="sub-header">RFD 分析結果</h3>', unsafe_allow_html=True)
                
                # RFD表の作成
                rfd_data = []
                peak_rfd = max([v for v in current_result['rfd_values'].values() if v is not None], default=1)
                
                for time_window, rfd_value in current_result['rfd_values'].items():
                    if rfd_value is not None:
                        relative_value = (rfd_value / peak_rfd) * 100 if peak_rfd > 0 else 0
                        rfd_data.append({
                            "時間区間": time_window,
                            "RFD値 (N/s)": f"{rfd_value:.2f}",
                            "相対値 (%)": f"{relative_value:.1f}"
                        })
                    else:
                        rfd_data.append({
                            "時間区間": time_window,
                            "RFD値 (N/s)": "N/A",
                            "相対値 (%)": "N/A"
                        })
                
                # データフレームとして表示
                rfd_df = pd.DataFrame(rfd_data)
                st.dataframe(rfd_df, use_container_width=True)
                
                st.markdown(f"**ピークRFD:** {peak_rfd:.2f} N/s")
                st.markdown(f"**Onset時の力:** {current_result['onset_force']:.2f} N")
            
            # グラフ表示
            st.markdown('<h3 class="sub-header">力-時間曲線</h3>', unsafe_allow_html=True)
            
            time_data = np.array(current_result['time_data'])
            filtered_force = np.array(current_result['filtered_force'])
            onset_index = current_result['onset_index']
            peak_force_index = current_result['peak_force_index']
            auto_onset_index = current_result['auto_onset_index']
            
            # Plotlyでグラフを作成
            fig = go.Figure()
            
            # フィルター済みの力データをプロット
            fig.add_trace(go.Scatter(
                x=time_data, 
                y=filtered_force,
                mode='lines',
                name='フィルター済み力データ',
                line=dict(color='blue', width=2)
            ))
            
            # ベースラインをプロット
            fig.add_trace(go.Scatter(
                x=[time_data[0], time_data[-1]],
                y=[current_result['baseline_mean'], current_result['baseline_mean']],
                mode='lines',
                name='ベースライン',
                line=dict(color='green', width=1, dash='dash')
            ))
            
            # Onset閾値ラインをプロット
            fig.add_trace(go.Scatter(
                x=[time_data[0], time_data[-1]],
                y=[current_result['threshold'], current_result['threshold']],
                mode='lines',
                name='Onset閾値',
                line=dict(color='orange', width=1, dash='dot')
            ))
            
            # 自動検出されたOnsetをプロット（手動調整時のみ表示）
            if current_result.get('manual_adjustment', False):
                fig.add_trace(go.Scatter(
                    x=[time_data[auto_onset_index]],
                    y=[filtered_force[auto_onset_index]],
                    mode='markers',
                    name='自動検出Onset',
                    marker=dict(color='gray', size=8, symbol='circle')
                ))
            
            # 現在のOnsetをプロット
            onset_color = 'red' if not current_result.get('manual_adjustment', False) else 'darkred'
            onset_name = 'Onset' if not current_result.get('manual_adjustment', False) else 'Onset (手動調整)'
            
            fig.add_trace(go.Scatter(
                x=[time_data[onset_index]],
                y=[filtered_force[onset_index]],
                mode='markers',
                name=onset_name,
                marker=dict(color=onset_color, size=10, symbol='circle')
            ))
            
            # ピーク力をプロット
            fig.add_trace(go.Scatter(
                x=[time_data[peak_force_index]],
                y=[filtered_force[peak_force_index]],
                mode='markers',
                name='ピーク力',
                marker=dict(color='darkred', size=10, symbol='star')
            ))
            
            # グラフレイアウトの設定
            fig.update_layout(
                title='力-時間曲線',
                xaxis_title='時間 (秒)',
                yaxis_title='力 (N)',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            
            # グラフを表示
            st.plotly_chart(fig, use_container_width=True)
            
            # CSV形式のダウンロードボタン
            st.markdown('<h3 class="sub-header">結果のエクスポート</h3>', unsafe_allow_html=True)
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                # 現在の試技の結果をCSVとしてダウンロード
                if st.button("現在の試技の結果をCSVとしてダウンロード", use_container_width=True):
                    # CSVファイルを作成
                    csv_data = []
                    csv_data.append(['項目', '値', '単位', '備考'])
                    
                    # 試技名（複数試技モードの場合）
                    if len(st.session_state['trial_names']) > 1:
                        csv_data.append(['試技名', st.session_state['trial_names'][st.session_state['selected_trial']], '', ''])
                    
                    # 測定の基本情報
                    csv_data.append(['━━━━━━━━━ 基本測定値 ━━━━━━━━━', '', '', ''])
                    csv_data.append(['測定日時', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '', ''])
                    csv_data.append(['安静時の平均値（体重相当）', f"{current_result['baseline_mean']:.2f}", 'N', ''])
                    csv_data.append(['Onset時点', f"{current_result['onset_time']:.3f}", '秒', ''])
                    csv_data.append(['Onset時の力', f"{current_result['onset_force']:.2f}", 'N', ''])
                    csv_data.append(['Peak Force', f"{current_result['peak_force']:.2f}", 'N', ''])
                    csv_data.append(['Net Peak Force（体重差し引き）', f"{current_result['net_peak_force']:.2f}", 'N', ''])
                    csv_data.append(['Peak Force タイミング', f"{current_result['peak_time']:.3f}", '秒', ''])
                    csv_data.append(['Onsetからピークまでの時間', f"{current_result['time_to_peak']:.3f}", '秒', '重要指標'])
                    
                    # 分析設定
                    csv_data.append([''])
                    csv_data.append(['━━━━━━━━━ 分析設定 ━━━━━━━━━', '', '', ''])
                    csv_data.append(['フィルター設定', f"{filter_freq} Hz, 4次 Butterworth", 'Hz', ''])
                    csv_data.append(['Onset閾値', f"ベースライン + {onset_threshold} SD", 'SD', ''])
                    csv_data.append(['サンプリングレート', f"{sampling_rate}", 'Hz', ''])
                    
                    # Onset調整情報
                    csv_data.append([''])
                    csv_data.append(['━━━━━━━━━ Onset検出情報 ━━━━━━━━━', '', '', ''])
                    csv_data.append(['自動検出Onset時間', f"{current_result['auto_onset_time']:.3f}", '秒', ''])
                    csv_data.append(['使用されたOnset時間', f"{current_result['onset_time']:.3f}", '秒', ''])
                    if current_result.get('manual_adjustment', False):
                        csv_data.append(['Onset調整状態', '手動調整済み', '', ''])
                    else:
                        csv_data.append(['Onset調整状態', '自動検出', '', ''])
                    
                    # RFD情報
                    csv_data.append([''])
                    csv_data.append(['━━━━━━━━━ RFD (Rate of Force Development) ━━━━━━━━━', '', '', ''])
                    csv_data.append(['時間区間', 'RFD値', '単位', '計算式'])
                    
                    # ピークRFDを計算
                    csv_data.append(['ピークRFD', f"{peak_rfd:.2f}", 'N/s', '最大RFD値'])
                    
                    # 各RFD値
                    for time_window, rfd_value in current_result['rfd_values'].items():
                        if rfd_value is not None:
                            csv_data.append([time_window, f"{rfd_value:.2f}", 'N/s', ''])
                        else:
                            csv_data.append([time_window, "データ不足で計算不可", 'N/s', ''])
                    
                    # CSVとしてエンコード
                    csv_string = io.StringIO()
                    writer = csv.writer(csv_string)
                    writer.writerows(csv_data)
                    
                    # ダウンロードリンクを作成
                    trial_name = st.session_state['trial_names'][st.session_state['selected_trial']]
                    filename = f"IMTP_分析結果_{trial_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    
                    st.download_button(
                        label="CSVファイルをダウンロード",
                        data=csv_string.getvalue(),
                        file_name=filename,
                        mime='text/csv',
                        use_container_width=True
                    )
            
            with export_col2:
                # 全試技の結果をCSVとしてダウンロード（複数試技がある場合）
                if len(st.session_state['trial_data']) > 1:
                    # 有効な結果があるか確認
                    valid_results = [r for r in st.session_state['trial_results'] if r is not None]
                    
                    if valid_results and st.button("全試技の結果をCSVとしてダウンロード", use_container_width=True):
                        # CSVファイルを作成
                        all_data = []
                        
                        # ヘッダー行
                        all_data.append(['試技名', '安静時平均 (N)', 'Onset時間 (s)', 'Peak Force (N)', 
                                        'Net Peak Force (N)', 'Time to Peak (s)', 'ピークRFD (N/s)',
                                        'RFD 0-50ms (N/s)', 'RFD 0-100ms (N/s)', 'RFD 0-150ms (N/s)', 
                                        'RFD 0-200ms (N/s)', 'RFD 0-250ms (N/s)', 'Onset調整状態', '自動検出Onset (s)'])
                        
                        # 各試技の結果を行として追加
                        for i, result in enumerate(st.session_state['trial_results']):
                            if result is not None:
                                if i < len(st.session_state['trial_names']):  # 試技名のインデックスチェック
                                    trial_name = st.session_state['trial_names'][i]
                                else:
                                    trial_name = f"試技 {i+1}"
                                    
                                row = [
                                    trial_name,
                                    f"{result['baseline_mean']:.2f}",
                                    f"{result['onset_time']:.3f}",
                                    f"{result['peak_force']:.2f}",
                                    f"{result['net_peak_force']:.2f}",
                                    f"{result['time_to_peak']:.3f}"
                                ]
                                
                                # ピークRFDを計算
                                peak_rfd = max([v for v in result['rfd_values'].values() if v is not None], default=0)
                                row.append(f"{peak_rfd:.2f}")
                                
                                # 各時間区間のRFD値を追加
                                for window in ['RFD 0-50ms', 'RFD 0-100ms', 'RFD 0-150ms', 'RFD 0-200ms', 'RFD 0-250ms']:
                                    if window in result['rfd_values'] and result['rfd_values'][window] is not None:
                                        row.append(f"{result['rfd_values'][window]:.2f}")
                                    else:
                                        row.append("N/A")
                                
                                # Onset調整状態を追加
                                if result.get('manual_adjustment', False):
                                    row.append("手動調整済み")
                                else:
                                    row.append("自動検出")
                                
                                # 自動検出Onset時間を追加
                                row.append(f"{result['auto_onset_time']:.3f}")
                                
                                all_data.append(row)
                        
                        # CSVとしてエンコード
                        csv_string = io.StringIO()
                        writer = csv.writer(csv_string)
                        writer.writerows(all_data)
                        
                        # ダウンロードリンクを作成
                        filename = f"IMTP_全試技分析結果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        
                        st.download_button(
                            label="全結果CSVファイルをダウンロード",
                            data=csv_string.getvalue(),
                            file_name=filename,
                            mime='text/csv',
                            use_container_width=True
                        )