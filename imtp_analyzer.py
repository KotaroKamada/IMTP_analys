import streamlit as st
import pandas as pd
import numpy as np
from scipy import signal
import plotly.graph_objects as go
import io
import csv
from datetime import datetime
import traceback

# ページ設定
st.set_page_config(
    page_title="IMTP分析アプリ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# スタイル設定
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
    .warning-text { color: #e74c3c; font-weight: bold; }
    .success-text { color: #2ecc71; font-weight: bold; }
    .onset-adjustment {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# タイトル
st.markdown('<h1 class="main-header">IMTP 分析ツール</h1>', unsafe_allow_html=True)

# セッション状態の初期化
def init_session_state():
    defaults = {
        'data': None,
        'selected_trial': 0,
        'trial_data': [],
        'trial_names': [],
        'trial_results': [],
        'manual_onset_adjustments': {},
        'analysis_completed': False,
        'current_view': 'input'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# サイドバー設定
st.sidebar.header("⚙️ 分析パラメータ")
onset_threshold = st.sidebar.slider("Onset閾値 (SD×):", 1.0, 10.0, 5.0, 0.1)
filter_freq = st.sidebar.slider("フィルター (Hz):", 10.0, 100.0, 50.0, 1.0)
sampling_rate = st.sidebar.number_input("サンプリングレート (Hz):", min_value=100, max_value=10000, value=1000, step=100)

# 解析関数群
def safe_apply_filter(force_data, filter_freq, sampling_rate):
    """安全なフィルター処理"""
    try:
        if len(force_data) < 6:
            st.warning("データ点数が少ないため、フィルター処理をスキップします")
            return force_data.copy()
        
        nyquist = 0.5 * sampling_rate
        cutoff = min(filter_freq / nyquist, 0.99)
        
        if cutoff <= 0:
            return force_data.copy()
        
        b, a = signal.butter(4, cutoff, btype='low')
        filtered_data = signal.filtfilt(b, a, force_data)
        
        if np.any(np.isnan(filtered_data)):
            st.warning("フィルター処理でNaN値が発生したため、元データを使用します")
            return force_data.copy()
        
        return filtered_data
    
    except Exception as e:
        st.warning(f"フィルター処理エラー: {str(e)}。元データを使用します。")
        return force_data.copy()

def safe_detect_onset(force_data, baseline_window, onset_threshold):
    """安全なOnset検出"""
    try:
        baseline_window = min(baseline_window, len(force_data) // 4)
        baseline_window = max(baseline_window, 10)
        
        if baseline_window >= len(force_data):
            baseline_window = len(force_data) // 2
        
        baseline_data = force_data[:baseline_window]
        baseline_mean = np.mean(baseline_data)
        baseline_std = np.std(baseline_data)
        
        if baseline_std == 0:
            baseline_std = 0.1
        
        threshold = baseline_mean + (baseline_std * onset_threshold)
        
        for i in range(baseline_window, len(force_data) - 5):
            if force_data[i] > threshold:
                if all(f > threshold for f in force_data[i:i+5]):
                    return i, baseline_mean, threshold
        
        return baseline_window, baseline_mean, threshold
    
    except Exception as e:
        st.warning(f"Onset検出エラー: {str(e)}。デフォルト値を使用します。")
        baseline_mean = np.mean(force_data[:min(100, len(force_data)//4)])
        return min(100, len(force_data)//4), baseline_mean, baseline_mean + 10

def safe_calculate_rfd(force_data, onset_index, sampling_rate):
    """安全なRFD計算"""
    try:
        rfd_results = {}
        time_windows = [50, 100, 150, 200, 250]
        
        if onset_index >= len(force_data):
            for window in time_windows:
                rfd_results[f"RFD 0-{window}ms"] = None
            return rfd_results
        
        force_at_onset = force_data[onset_index]
        
        for window in time_windows:
            try:
                points = int(window * sampling_rate / 1000)
                if onset_index + points < len(force_data):
                    force_change = force_data[onset_index + points] - force_at_onset
                    rfd = force_change / (window / 1000)
                    
                    if not np.isnan(rfd) and not np.isinf(rfd):
                        rfd_results[f"RFD 0-{window}ms"] = float(rfd)
                    else:
                        rfd_results[f"RFD 0-{window}ms"] = None
                else:
                    rfd_results[f"RFD 0-{window}ms"] = None
            except Exception:
                rfd_results[f"RFD 0-{window}ms"] = None
        
        return rfd_results
    
    except Exception as e:
        st.warning(f"RFD計算エラー: {str(e)}")
        rfd_results = {}
        for window in [50, 100, 150, 200, 250]:
            rfd_results[f"RFD 0-{window}ms"] = None
        return rfd_results

def analyze_trial_safe(time_data, force_data, filter_freq, onset_threshold, sampling_rate, baseline_window, manual_onset_time=None):
    """安全な試技分析"""
    try:
        # データ検証
        if len(time_data) < 50 or len(force_data) < 50:
            raise ValueError("データ点数が不足しています（最低50点必要）")
        
        # NaN値チェックと除去
        valid_indices = ~(np.isnan(time_data) | np.isnan(force_data))
        if not np.all(valid_indices):
            time_data = time_data[valid_indices]
            force_data = force_data[valid_indices]
            st.warning("NaN値を除去しました")
        
        if len(time_data) < 50:
            raise ValueError("有効なデータ点数が不足しています")
        
        # フィルター適用
        filtered_force = safe_apply_filter(force_data, filter_freq, sampling_rate)
        
        # 自動Onset検出
        auto_onset_index, baseline_mean, threshold = safe_detect_onset(filtered_force, baseline_window, onset_threshold)
        auto_onset_time = auto_onset_index / sampling_rate
        
        # Onset設定
        if manual_onset_time is not None:
            onset_index = int(manual_onset_time * sampling_rate)
            onset_time = manual_onset_time
        else:
            onset_index = auto_onset_index
            onset_time = auto_onset_time
        
        onset_index = max(0, min(onset_index, len(filtered_force) - 1))
        
        # ピーク検出
        if onset_index < len(filtered_force) - 1:
            peak_force = np.max(filtered_force[onset_index:])
            peak_force_index = np.argmax(filtered_force[onset_index:]) + onset_index
        else:
            peak_force = np.max(filtered_force)
            peak_force_index = np.argmax(filtered_force)
        
        onset_force = filtered_force[onset_index]
        time_to_peak = (peak_force_index - onset_index) / sampling_rate
        
        # RFD計算
        rfd_values = safe_calculate_rfd(filtered_force, onset_index, sampling_rate)
        
        return {
            'baseline_mean': float(baseline_mean),
            'threshold': float(threshold),
            'auto_onset_index': int(auto_onset_index),
            'auto_onset_time': float(auto_onset_time),
            'onset_index': int(onset_index),
            'onset_time': float(onset_time),
            'onset_force': float(onset_force),
            'peak_force': float(peak_force),
            'peak_force_index': int(peak_force_index),
            'peak_time': float(peak_force_index / sampling_rate),
            'net_peak_force': float(peak_force - baseline_mean),
            'time_to_peak': float(time_to_peak),
            'rfd_values': rfd_values,
            'filtered_force': filtered_force.tolist(),
            'time_data': time_data.tolist(),
            'manual_adjustment': manual_onset_time is not None
        }
    
    except Exception as e:
        st.error(f"分析エラー: {str(e)}")
        return None

# データ入力セクション
st.markdown('<h2 class="sub-header">📂 データ入力</h2>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("ファイルをアップロード", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        with st.spinner('ファイル読み込み中...'):
            # ファイル読み込み
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            # データ前処理
            data.columns = [str(col).strip().replace(' ', '_') for col in data.columns]
            
            # 数値列のみを処理
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = data[numeric_columns].fillna(method='ffill')
            
            # 複数試技判定
            if len(data.columns) > 2:
                # 複数試技
                time_column = data.columns[0]
                time_data_base = data[time_column].values
                trial_data = []
                trial_names = []
                
                for col in data.columns[1:]:
                    trial_names.append(col)
                    trial_df = pd.DataFrame({
                        'time': time_data_base,
                        'force': data[col].values
                    })
                    trial_data.append(trial_df)
                
                st.session_state['trial_data'] = trial_data
                st.session_state['trial_names'] = trial_names
                st.session_state['trial_results'] = [None] * len(trial_data)
                st.session_state['data'] = trial_data[0]
                st.success(f"✅ {len(trial_data)}試技を読み込みました")
            else:
                # 単一試技
                st.session_state['data'] = data
                st.session_state['trial_data'] = [data]
                st.session_state['trial_names'] = ["試技1"]
                st.session_state['trial_results'] = [None]
                st.success("✅ 単一試技を読み込みました")
            
            st.session_state['selected_trial'] = 0
            
    except Exception as e:
        st.error(f"❌ ファイル読み込みエラー: {str(e)}")

# メイン処理
if st.session_state['data'] is not None:
    
    # ビュー切り替え
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("📊 データ入力・分析", use_container_width=True):
            st.session_state['current_view'] = 'input'
            st.rerun()
    
    with col2:
        has_results = (st.session_state['trial_results'] and 
                      st.session_state['selected_trial'] < len(st.session_state['trial_results']) and
                      st.session_state['trial_results'][st.session_state['selected_trial']] is not None)
        if has_results:
            if st.button("📈 分析結果・調整", use_container_width=True):
                st.session_state['current_view'] = 'results'
                st.rerun()
        else:
            st.button("📈 分析結果・調整", disabled=True, use_container_width=True)
    
    with col3:
        current_view_text = "データ入力・分析" if st.session_state['current_view'] == 'input' else "分析結果・調整"
        st.markdown(f"**現在のビュー:** {current_view_text}")
    
    st.divider()
    
    # ビュー表示
    if st.session_state['current_view'] == 'input':
        # === データ入力・分析ビュー ===
        
        # データプレビュー
        st.markdown('<h3 class="sub-header">📋 データプレビュー</h3>', unsafe_allow_html=True)
        st.dataframe(st.session_state['data'].head(5), use_container_width=True)
        
        # 試技選択
        if len(st.session_state['trial_names']) > 1:
            st.markdown('<h3 class="sub-header">🎯 試技選択</h3>', unsafe_allow_html=True)
            selected_trial_name = st.selectbox("試技:", st.session_state['trial_names'],
                                              index=st.session_state['selected_trial'])
            selected_trial_index = st.session_state['trial_names'].index(selected_trial_name)
            if selected_trial_index != st.session_state['selected_trial']:
                st.session_state['selected_trial'] = selected_trial_index
                st.session_state['data'] = st.session_state['trial_data'][selected_trial_index]
                st.rerun()
        
        # 列選択
        st.markdown('<h3 class="sub-header">🔧 列選択</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            time_column = st.selectbox("時間列:", st.session_state['data'].columns.tolist(), index=0)
        
        with col2:
            force_column = st.selectbox("力列:", st.session_state['data'].columns.tolist(),
                                       index=min(1, len(st.session_state['data'].columns)-1))
        
        # 分析実行
        st.markdown('<h3 class="sub-header">🚀 分析実行</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("現在の試技を分析", type="primary", use_container_width=True):
                try:
                    with st.spinner('分析中...'):
                        time_data = st.session_state['data'][time_column].values
                        force_data = st.session_state['data'][force_column].values
                        
                        baseline_window = int(sampling_rate)
                        trial_key = f"{st.session_state['selected_trial']}_{st.session_state['trial_names'][st.session_state['selected_trial']]}"
                        manual_onset = st.session_state['manual_onset_adjustments'].get(trial_key, None)
                        
                        result = analyze_trial_safe(
                            time_data, force_data, filter_freq, onset_threshold,
                            sampling_rate, baseline_window, manual_onset_time=manual_onset
                        )
                        
                        if result is not None:
                            # 結果保存の安全化
                            while len(st.session_state['trial_results']) <= st.session_state['selected_trial']:
                                st.session_state['trial_results'].append(None)
                            
                            st.session_state['trial_results'][st.session_state['selected_trial']] = result
                            st.session_state['analysis_completed'] = True
                            st.session_state['current_view'] = 'results'
                            st.success('✅ 分析完了！結果ビューに切り替わります。')
                            st.rerun()
                        else:
                            st.error("❌ 分析に失敗しました")
                
                except Exception as e:
                    st.error(f"❌ 分析エラー: {str(e)}")
        
        with col2:
            if len(st.session_state['trial_data']) > 1:
                if st.button("全試技を分析", use_container_width=True):
                    try:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        success_count = 0
                        error_count = 0
                        
                        for i, trial_df in enumerate(st.session_state['trial_data']):
                            try:
                                # 進捗更新
                                progress = (i + 1) / len(st.session_state['trial_data'])
                                progress_bar.progress(progress)
                                status_text.text(f"分析中: {i+1}/{len(st.session_state['trial_data'])} - {st.session_state['trial_names'][i]}")
                                
                                # データ取得
                                columns = list(trial_df.columns)
                                if len(columns) < 2:
                                    error_count += 1
                                    continue
                                
                                time_data = trial_df[columns[0]].values
                                force_data = trial_df[columns[1]].values
                                
                                baseline_window = int(sampling_rate)
                                trial_key = f"{i}_{st.session_state['trial_names'][i]}"
                                manual_onset = st.session_state['manual_onset_adjustments'].get(trial_key, None)
                                
                                result = analyze_trial_safe(
                                    time_data, force_data, filter_freq, onset_threshold,
                                    sampling_rate, baseline_window, manual_onset_time=manual_onset
                                )
                                
                                # 結果保存の安全化
                                while len(st.session_state['trial_results']) <= i:
                                    st.session_state['trial_results'].append(None)
                                
                                if result is not None:
                                    st.session_state['trial_results'][i] = result
                                    success_count += 1
                                else:
                                    st.session_state['trial_results'][i] = None
                                    error_count += 1
                            
                            except Exception as e:
                                st.warning(f"試技 {i+1} でエラー: {str(e)}")
                                error_count += 1
                                
                                while len(st.session_state['trial_results']) <= i:
                                    st.session_state['trial_results'].append(None)
                                st.session_state['trial_results'][i] = None
                        
                        progress_bar.progress(1.0)
                        status_text.text("分析完了")
                        
                        if success_count > 0:
                            st.session_state['current_view'] = 'results'
                            st.session_state['analysis_completed'] = True
                            
                            if error_count == 0:
                                st.success(f'✅ 全 {success_count} 試技の分析が完了しました！')
                            else:
                                st.warning(f'⚠️ 分析完了: {success_count} 試技成功, {error_count} 試技失敗')
                            
                            st.rerun()
                        else:
                            st.error("❌ 全試技の分析に失敗しました")
                    
                    except Exception as e:
                        st.error(f"❌ 全試技分析エラー: {str(e)}")
    
    elif st.session_state['current_view'] == 'results':
        # === 分析結果・調整ビュー ===
        
        # 結果の存在確認
        if (st.session_state['selected_trial'] >= len(st.session_state['trial_results']) or 
            st.session_state['trial_results'][st.session_state['selected_trial']] is None):
            
            st.warning("分析結果がありません。先にデータ入力・分析ビューで分析を実行してください。")
            if st.button("📊 データ入力・分析ビューに戻る"):
                st.session_state['current_view'] = 'input'
                st.rerun()
        else:
            # 試技選択
            if len(st.session_state['trial_names']) > 1:
                st.markdown('<h3 class="sub-header">🎯 試技選択</h3>', unsafe_allow_html=True)
                selected_trial_name = st.selectbox("試技:", st.session_state['trial_names'],
                                                  index=st.session_state['selected_trial'], key="results_trial")
                selected_trial_index = st.session_state['trial_names'].index(selected_trial_name)
                if selected_trial_index != st.session_state['selected_trial']:
                    st.session_state['selected_trial'] = selected_trial_index
                    # *** バグ修正：選択した試技のデータを更新 ***
                    st.session_state['data'] = st.session_state['trial_data'][selected_trial_index]
                    st.rerun()
            
            # 現在の結果を取得
            current_result = st.session_state['trial_results'][st.session_state['selected_trial']]
            
            st.markdown('<h2 class="sub-header">📊 分析結果</h2>', unsafe_allow_html=True)
            
            # Onset調整セクション
            st.markdown('<div class="onset-adjustment">', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">🎯 Onset調整</h3>', unsafe_allow_html=True)
            
            trial_key = f"{st.session_state['selected_trial']}_{st.session_state['trial_names'][st.session_state['selected_trial']]}"
            
            onset_col1, onset_col2, onset_col3 = st.columns([2, 2, 1])
            
            with onset_col1:
                auto_onset_time = current_result['auto_onset_time']
                current_onset_time = current_result['onset_time']
                is_manual = current_result.get('manual_adjustment', False)
                
                st.markdown(f"**自動検出Onset:** {auto_onset_time:.3f} 秒")
                if is_manual:
                    st.markdown(f"**現在のOnset:** <span class='warning-text'>{current_onset_time:.3f} 秒 (手動調整)</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**現在のOnset:** <span class='success-text'>{current_onset_time:.3f} 秒 (自動検出)</span>", unsafe_allow_html=True)
            
            with onset_col2:
                # 調整値の初期化
                adjustment_key = f"temp_onset_{trial_key}"
                if adjustment_key not in st.session_state:
                    st.session_state[adjustment_key] = current_onset_time
                
                new_onset_value = st.number_input(
                    "Onset調整 (秒):",
                    value=st.session_state[adjustment_key],
                    step=0.001,
                    format="%.3f",
                    key=f"onset_input_{trial_key}"
                )
                
                # 差分表示
                diff_ms = (new_onset_value - auto_onset_time) * 1000
                if abs(diff_ms) > 0.5:
                    st.markdown(f"**調整量:** {diff_ms:+.1f} ms")
            
            with onset_col3:
                # 適用ボタン
                if st.button("🔄 適用", key=f"apply_{trial_key}", type="primary"):
                    try:
                        with st.spinner('調整適用中...'):
                            st.session_state['manual_onset_adjustments'][trial_key] = new_onset_value
                            
                            # *** バグ修正：現在選択している試技のデータを使用 ***
                            current_trial_data = st.session_state['trial_data'][st.session_state['selected_trial']]
                            time_data = np.array(current_result['time_data'])
                            force_data_raw = current_trial_data.iloc[:, 1].values  # 力データ列（2列目）を取得
                            baseline_window = int(sampling_rate)
                            
                            result = analyze_trial_safe(
                                time_data, force_data_raw, filter_freq, onset_threshold,
                                sampling_rate, baseline_window, manual_onset_time=new_onset_value
                            )
                            
                            if result is not None:
                                st.session_state['trial_results'][st.session_state['selected_trial']] = result
                                st.success("✅ 調整が適用されました！")
                                st.rerun()
                            else:
                                st.error("❌ 調整適用に失敗しました")
                    
                    except Exception as e:
                        st.error(f"❌ 調整適用エラー: {str(e)}")
                
                # リセットボタン
                if st.button("↩️ リセット", key=f"reset_{trial_key}"):
                    try:
                        with st.spinner('リセット中...'):
                            if trial_key in st.session_state['manual_onset_adjustments']:
                                del st.session_state['manual_onset_adjustments'][trial_key]
                            
                            st.session_state[adjustment_key] = auto_onset_time
                            
                            # *** バグ修正：現在選択している試技のデータを使用 ***
                            current_trial_data = st.session_state['trial_data'][st.session_state['selected_trial']]
                            time_data = np.array(current_result['time_data'])
                            force_data_raw = current_trial_data.iloc[:, 1].values  # 力データ列（2列目）を取得
                            baseline_window = int(sampling_rate)
                            
                            result = analyze_trial_safe(
                                time_data, force_data_raw, filter_freq, onset_threshold,
                                sampling_rate, baseline_window, manual_onset_time=None
                            )
                            
                            if result is not None:
                                st.session_state['trial_results'][st.session_state['selected_trial']] = result
                                st.success("✅ 自動検出に戻しました！")
                                st.rerun()
                            else:
                                st.error("❌ リセットに失敗しました")
                    
                    except Exception as e:
                        st.error(f"❌ リセットエラー: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 結果表示
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.markdown('<h4 class="sub-header">📋 基本測定値</h4>', unsafe_allow_html=True)
                
                if len(st.session_state['trial_names']) > 1:
                    st.markdown(f"**試技:** {st.session_state['trial_names'][st.session_state['selected_trial']]}")
                
                st.markdown("**━━━━━━━━━ 測定結果 ━━━━━━━━━**")
                st.markdown(f"**安静時平均値:** {current_result['baseline_mean']:.2f} N")
                st.markdown(f"**Onset時点:** {current_result['onset_time']:.3f} 秒")
                st.markdown(f"**Onset時の力:** {current_result['onset_force']:.2f} N")
                st.markdown(f"**Peak Force:** {current_result['peak_force']:.2f} N")
                st.markdown(f"**Net Peak Force:** {current_result['net_peak_force']:.2f} N")
                st.markdown(f"**Peak時点:** {current_result['peak_time']:.3f} 秒")
                st.markdown(f"**Time to Peak:** <span class='warning-text'>{current_result['time_to_peak']:.3f} 秒</span>", unsafe_allow_html=True)
                
                st.markdown("**━━━━━━━━━ 分析設定 ━━━━━━━━━**")
                st.markdown(f"**フィルター:** {filter_freq:.1f} Hz, 4次 Butterworth")
                st.markdown(f"**Onset閾値:** ベースライン + {onset_threshold:.1f} SD")
                st.markdown(f"**サンプリングレート:** {sampling_rate} Hz")
                
            with result_col2:
                st.markdown('<h4 class="sub-header">📈 RFD分析結果</h4>', unsafe_allow_html=True)
                
                # RFD表の作成
                rfd_data = []
                rfd_values = current_result['rfd_values']
                valid_rfd_values = [v for v in rfd_values.values() if v is not None]
                peak_rfd = max(valid_rfd_values) if valid_rfd_values else 0
                
                for time_window, rfd_value in rfd_values.items():
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
                if rfd_data:
                    rfd_df = pd.DataFrame(rfd_data)
                    st.dataframe(rfd_df, use_container_width=True)
                
                st.markdown(f"**ピークRFD:** {peak_rfd:.2f} N/s")
                st.markdown(f"**Onset時の力:** {current_result['onset_force']:.2f} N")
            
            # グラフ表示
            st.markdown('<h4 class="sub-header">📊 力-時間曲線</h4>', unsafe_allow_html=True)
            
            try:
                time_data = np.array(current_result['time_data'])
                filtered_force = np.array(current_result['filtered_force'])
                
                # データ間引き（パフォーマンス向上）
                step = max(1, len(time_data) // 2000)
                time_plot = time_data[::step]
                force_plot = filtered_force[::step]
                
                fig = go.Figure()
                
                # メインデータ
                fig.add_trace(go.Scatter(
                    x=time_plot, y=force_plot,
                    mode='lines', name='フィルター済み力データ',
                    line=dict(color='blue', width=2)
                ))
                
                # ベースライン
                fig.add_trace(go.Scatter(
                    x=[time_data[0], time_data[-1]],
                    y=[current_result['baseline_mean'], current_result['baseline_mean']],
                    mode='lines', name='ベースライン',
                    line=dict(color='green', width=1, dash='dash')
                ))
                
                # Onset閾値
                fig.add_trace(go.Scatter(
                    x=[time_data[0], time_data[-1]],
                    y=[current_result['threshold'], current_result['threshold']],
                    mode='lines', name='Onset閾値',
                    line=dict(color='orange', width=1, dash='dot')
                ))
                
                # 自動検出Onset（手動調整時）
                if current_result.get('manual_adjustment', False):
                    auto_onset_index = current_result['auto_onset_index']
                    if auto_onset_index < len(filtered_force):
                        fig.add_trace(go.Scatter(
                            x=[time_data[auto_onset_index]],
                            y=[filtered_force[auto_onset_index]],
                            mode='markers', name='自動検出Onset',
                            marker=dict(color='gray', size=8, symbol='circle')
                        ))
                
                # 現在のOnset
                onset_index = current_result['onset_index']
                if onset_index < len(filtered_force):
                    onset_color = 'red' if not current_result.get('manual_adjustment', False) else 'darkred'
                    onset_name = 'Onset' if not current_result.get('manual_adjustment', False) else 'Onset (手動調整)'
                    
                    fig.add_trace(go.Scatter(
                        x=[time_data[onset_index]],
                        y=[filtered_force[onset_index]],
                        mode='markers', name=onset_name,
                        marker=dict(color=onset_color, size=10, symbol='circle')
                    ))
                
                # ピーク力
                peak_force_index = current_result['peak_force_index']
                if peak_force_index < len(filtered_force):
                    fig.add_trace(go.Scatter(
                        x=[time_data[peak_force_index]],
                        y=[filtered_force[peak_force_index]],
                        mode='markers', name='ピーク力',
                        marker=dict(color='darkred', size=10, symbol='star')
                    ))
                
                fig.update_layout(
                    title='力-時間曲線',
                    xaxis_title='時間 (秒)',
                    yaxis_title='力 (N)',
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                    height=500,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"グラフ表示エラー: {str(e)}")
            
            # エクスポート
            st.markdown('<h4 class="sub-header">💾 結果エクスポート</h4>', unsafe_allow_html=True)
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                if st.button("📥 現在の結果をCSV保存", use_container_width=True):
                    try:
                        csv_data = []
                        csv_data.append(['項目', '値', '単位', '備考'])
                        
                        if len(st.session_state['trial_names']) > 1:
                            csv_data.append(['試技名', st.session_state['trial_names'][st.session_state['selected_trial']], '', ''])
                        
                        csv_data.append(['━━━━━━━━━ 基本測定値 ━━━━━━━━━', '', '', ''])
                        csv_data.append(['測定日時', datetime.now().strftime("%Y-%m-%d %H:%M:%S"), '', ''])
                        csv_data.append(['安静時平均値', f"{current_result['baseline_mean']:.2f}", 'N', ''])
                        csv_data.append(['Onset時点', f"{current_result['onset_time']:.3f}", '秒', ''])
                        csv_data.append(['Onset時の力', f"{current_result['onset_force']:.2f}", 'N', ''])
                        csv_data.append(['Peak Force', f"{current_result['peak_force']:.2f}", 'N', ''])
                        csv_data.append(['Net Peak Force', f"{current_result['net_peak_force']:.2f}", 'N', ''])
                        csv_data.append(['Peak時点', f"{current_result['peak_time']:.3f}", '秒', ''])
                        csv_data.append(['Time to Peak', f"{current_result['time_to_peak']:.3f}", '秒', '重要指標'])
                        
                        csv_data.append(['━━━━━━━━━ 分析設定 ━━━━━━━━━', '', '', ''])
                        csv_data.append(['フィルター設定', f"{filter_freq:.1f} Hz, 4次 Butterworth", 'Hz', ''])
                        csv_data.append(['Onset閾値', f"ベースライン + {onset_threshold:.1f} SD", 'SD', ''])
                        csv_data.append(['サンプリングレート', f"{sampling_rate}", 'Hz', ''])
                        
                        csv_data.append(['━━━━━━━━━ Onset検出情報 ━━━━━━━━━', '', '', ''])
                        csv_data.append(['自動検出Onset', f"{current_result['auto_onset_time']:.3f}", '秒', ''])
                        csv_data.append(['使用Onset', f"{current_result['onset_time']:.3f}", '秒', ''])
                        csv_data.append(['調整状態', '手動調整' if current_result.get('manual_adjustment', False) else '自動検出', '', ''])
                        
                        csv_data.append(['━━━━━━━━━ RFD結果 ━━━━━━━━━', '', '', ''])
                        csv_data.append(['ピークRFD', f"{peak_rfd:.2f}", 'N/s', '最大RFD値'])
                        
                        for time_window, rfd_value in current_result['rfd_values'].items():
                            if rfd_value is not None:
                                csv_data.append([time_window, f"{rfd_value:.2f}", 'N/s', ''])
                            else:
                                csv_data.append([time_window, "N/A", 'N/s', 'データ不足'])
                        
                        csv_string = io.StringIO()
                        writer = csv.writer(csv_string)
                        writer.writerows(csv_data)
                        
                        trial_name = st.session_state['trial_names'][st.session_state['selected_trial']]
                        filename = f"IMTP_分析結果_{trial_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        
                        st.download_button(
                            label="CSVダウンロード",
                            data=csv_string.getvalue(),
                            file_name=filename,
                            mime='text/csv',
                            use_container_width=True
                        )
                    
                    except Exception as e:
                        st.error(f"CSV作成エラー: {str(e)}")
            
            with export_col2:
                if len(st.session_state['trial_data']) > 1:
                    valid_results = [r for r in st.session_state['trial_results'] if r is not None]
                    if valid_results and st.button("📥 全結果をCSV保存", use_container_width=True):
                        try:
                            all_data = []
                            all_data.append(['試技名', '安静時平均(N)', 'Onset時間(s)', 'Peak Force(N)', 
                                            'Net Peak Force(N)', 'Time to Peak(s)', 'ピークRFD(N/s)',
                                            'RFD 0-50ms(N/s)', 'RFD 0-100ms(N/s)', 'RFD 0-150ms(N/s)', 
                                            'RFD 0-200ms(N/s)', 'RFD 0-250ms(N/s)', 'Onset調整状態', '自動検出Onset(s)'])
                            
                            for i, result in enumerate(st.session_state['trial_results']):
                                if result is not None:
                                    trial_name = st.session_state['trial_names'][i] if i < len(st.session_state['trial_names']) else f"試技{i+1}"
                                    
                                    row = [
                                        trial_name,
                                        f"{result['baseline_mean']:.2f}",
                                        f"{result['onset_time']:.3f}",
                                        f"{result['peak_force']:.2f}",
                                        f"{result['net_peak_force']:.2f}",
                                        f"{result['time_to_peak']:.3f}"
                                    ]
                                    
                                    # ピークRFD
                                    valid_rfd = [v for v in result['rfd_values'].values() if v is not None]
                                    peak_rfd_val = max(valid_rfd) if valid_rfd else 0
                                    row.append(f"{peak_rfd_val:.2f}")
                                    
                                    # 各RFD値
                                    for window in ['RFD 0-50ms', 'RFD 0-100ms', 'RFD 0-150ms', 'RFD 0-200ms', 'RFD 0-250ms']:
                                        if window in result['rfd_values'] and result['rfd_values'][window] is not None:
                                            row.append(f"{result['rfd_values'][window]:.2f}")
                                        else:
                                            row.append("N/A")
                                    
                                    # 調整状態
                                    row.append("手動調整" if result.get('manual_adjustment', False) else "自動検出")
                                    row.append(f"{result['auto_onset_time']:.3f}")
                                    
                                    all_data.append(row)
                            
                            csv_string = io.StringIO()
                            writer = csv.writer(csv_string)
                            writer.writerows(all_data)
                            
                            filename = f"IMTP_全試技分析結果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                            
                            st.download_button(
                                label="全結果CSVダウンロード",
                                data=csv_string.getvalue(),
                                file_name=filename,
                                mime='text/csv',
                                use_container_width=True
                            )
                        
                        except Exception as e:
                            st.error(f"全結果CSV作成エラー: {str(e)}")

# フッター
st.markdown("---")
st.markdown("🔬 **IMTP分析ツール** - 安定版 v1.0")
st.markdown("📧 不具合やご要望がございましたら、開発者までお知らせください。")