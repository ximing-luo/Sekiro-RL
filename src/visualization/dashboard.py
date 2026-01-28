"""
模块用途：训练过程的实时与历史阈值/指标可视化界面（Streamlit）。

包含：
- 函数：load_latest()
- 视图：当前步、历史趋势、阈值矩阵

边界：
- 负责：读取 latest.json / train_metrics.csv 并展示
- 不负责：训练与环境逻辑、文件写入
"""
import json
import os
import sys
import time
import streamlit as st
import pandas as pd
import altair as alt

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import configs.config as config
from src.envs.tasks.sekiro.action_map import ACTION_LABELS
from src.envs.tasks.sekiro.action_map import ACTION_FUNC_MAP, action_count, no_op_index
try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AR = True
except Exception:
    _HAS_AR = False

# 动作短标签来源：date/actions_map.py
EVENT_LABELS = ['自身死亡','Boss死亡','自身掉血','自身回血','自身血量过低','Boss掉血','自身架势上升','Boss架势上升','Boss架势过低']

LOG_DIR = config.LOG_DIR
JSON_PATH = os.path.join(LOG_DIR, 'latest.json')
CSV_PATH = os.path.join(LOG_DIR, 'train_metrics.csv')

st.set_page_config(page_title='RL Thresholds Viewer', layout='wide')
st.title('RL Thresholds Viewer')

# 启动一致性检查
_act_map_len = len(ACTION_FUNC_MAP)
_act_dim = int(action_count())
if _act_map_len != _act_dim:
    st.sidebar.error(f"动作映射数量({_act_map_len})与统一动作维度({_act_dim})不一致")
if len(ACTION_LABELS) != _act_dim:
    st.sidebar.warning(f"动作标签数量({len(ACTION_LABELS)})与统一动作维度({_act_dim})不一致")

def load_latest():
    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

placeholder_info = st.empty()
cols = st.columns(3)

autorefresh = st.sidebar.slider('刷新间隔（秒）', 0.1, 5.0, 0.8)
view_mode = st.sidebar.radio('视图模式', ['当前步', '历史趋势'], index=0)

# 添加历史趋势刷新控制
if view_mode == '历史趋势':
    history_refresh_mode = st.sidebar.radio('历史趋势刷新模式', ['手动', '每2000步', '每3分钟'], index=0)
    if history_refresh_mode == '每2000步':
        history_refresh_interval = 2000
    elif history_refresh_mode == '每3分钟':
        history_refresh_interval = 180  # 秒间隔
    else:
        history_refresh_interval = None
else:
    history_refresh_interval = None

if _HAS_AR:
    if view_mode == '历史趋势':
        if history_refresh_mode == '每3分钟':
            st_autorefresh(interval=180000, key='history_refresh')
        elif history_refresh_mode == '每2000步':
            st_autorefresh(interval=5000, key='data_refresh')
    else:
        st_autorefresh(interval=int(autorefresh * 1000), key='data_refresh')
    st.session_state['enable_auto_checkbox'] = False
else:
    enable_auto = st.sidebar.checkbox('启用自动刷新', value=True, key='enable_auto_checkbox')
    if st.sidebar.button('手动刷新'):
        st.rerun()

st.sidebar.write(f'日志路径: {JSON_PATH}')

# 读取坐标范围（使用会话状态，不在此处渲染控件以便将控件放在底部）
def _ensure_axis_defaults():
            defaults = {
                'y_min_current': -20.0,
                'y_max_current': 20.0,
                'y_min_event_current': 0.0,
                'y_max_event_current': 2.0,
                'y_min_history': -20.0,
                'y_max_history': 20.0,
                'y_min_event_history': 0.0,
                'y_max_event_history': 2.0,
                'current_cols': 3,
                'current_chart_width': 450,
                'current_chart_height': 560,
                'event_chart_width': 380,
                'event_chart_height': 560,
                'label_mode': '数字',
                'use_auto_bar_ratio': True,
                'current_bar_fill_ratio': 0.66,
                'last_history_step': 0,  # 用于控制历史趋势刷新
                'use_auto_line_ratio_hist': True,
                'history_line_fill_ratio': 0.66,
                'center_mode_hist': '中点居中',
                'recent_actions': [],
                'recent_max': 15,
                'last_step_recent': -1,
                'recent_font_size': 20,
            }
            for k, v in defaults.items():
                if k not in st.session_state:
                    st.session_state[k] = v

_ensure_axis_defaults()
y_min = st.session_state['y_min_current']
y_max = st.session_state['y_max_current']
event_y_min = st.session_state['y_min_event_current']
event_y_max = st.session_state['y_max_event_current']
hist_y_min = st.session_state['y_min_history']
hist_y_max = st.session_state['y_max_history']
event_hist_y_min = st.session_state['y_min_event_history']
event_hist_y_max = st.session_state['y_max_event_history']

data = load_latest()
if not data:
    placeholder_info.warning('等待训练写入 latest.json ...')
else:
    step = data.get('step', 0)
    action = data.get('action', 0)
    events_feedback = data.get('events_feedback', [])
    events_list = data.get('events', [])
    raw_reward = data.get('raw_reward', None)
    q_values = data.get('q_values', [])
    action_recent_counts = data.get('action_recent_counts', [])
    adv_values = data.get('adv_values', [])
    adv_values_shrink = data.get('adv_values_shrink', [])
    state_value = data.get('state_value', None)
    noop_1s_count = data.get('noop_1s_count', None)
    epsilon = data.get('epsilon', None)

    try:
        q_values = [float(x) for x in q_values]
    except Exception:
        q_values = list(q_values)

    n_q = len(q_values)

    if view_mode == '当前步':
        mcols = st.columns(6)
        def _metric(col, title, value):
            col.markdown(f"<div style='text-align:center'><div style='font-size:14px;color:#666'>{title}</div><div style='font-size:28px;font-weight:600'>{value}</div></div>", unsafe_allow_html=True)
        act_label = ACTION_LABELS[action] if isinstance(action, int) and action < len(ACTION_LABELS) else str(action)
        _metric(mcols[0], 'Step', step)
        _metric(mcols[1], 'Action', f"{action} {act_label}")
        _metric(mcols[2], 'Events', ','.join([str(int(e)) for e in (events_list or [])]) if isinstance(events_list, list) and len(events_list) > 0 else '-')
        _metric(mcols[3], 'Reward', f"{float(raw_reward):.3f}" if raw_reward is not None else "-")
        _metric(mcols[4], 'No-op/1s', int(noop_1s_count) if noop_1s_count is not None else 0)
        # fps_val = data.get('fps', None)
        # _metric(mcols[5], 'FPS', f"{float(fps_val):.2f}" if fps_val is not None else '-')
        if epsilon is not None:
             _metric(mcols[5], 'Epsilon', f"{float(epsilon):.3f}")
        else:
             fps_val = data.get('fps', None)
             _metric(mcols[5], 'FPS', f"{float(fps_val):.2f}" if fps_val is not None else '-')

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        
        if step != st.session_state.get('last_step_recent', -1):
            st.session_state['last_step_recent'] = step
            _ra = list(st.session_state.get('recent_actions', []))
            _ra.insert(0, {'idx': action, 'label': act_label})
            _ra = _ra[:int(st.session_state.get('recent_max', 15))]
            st.session_state['recent_actions'] = _ra
        # 只显示 Q 值条形图
        label_mode = st.session_state.get('label_mode', '数字')
        idxs = list(range(len(q_values)))
        df_q = pd.DataFrame({
            'index': idxs,
            'Q': q_values,
        })
        if label_mode.startswith('中文'):
            labels_arr = '[' + ','.join([f'"{ACTION_LABELS[i] if i < len(ACTION_LABELS) else str(i)}"' for i in idxs]) + ']'
            x_axis = alt.X('index:O', axis=alt.Axis(title=None, labelAngle=(90 if label_mode.endswith('竖排') else 0), labelFontSize=12, labelExpr=f'{labels_arr}[datum.value]'))
        else:
            x_axis = alt.X('index:O', axis=alt.Axis(title=None, labelAngle=0, labelFontSize=12))
        _use_auto_ratio = st.session_state.get('use_auto_bar_ratio', True)
        _fill_ratio = float(st.session_state.get('current_bar_fill_ratio', 0.66))
        _mx_q = (max(q_values) if isinstance(q_values, list) and len(q_values) > 0 else 0.0)
        _mn_q = (min(q_values) if isinstance(q_values, list) and len(q_values) > 0 else 0.0)
        _dom_q = [y_min, y_max]
        if _use_auto_ratio:
            _posD = (_mx_q / _fill_ratio) if _mx_q > 0 else 1.0
            _negD = (abs(_mn_q) / _fill_ratio) if _mn_q < 0 else 0.0
            _dom_q = ([-_negD, _posD] if _mn_q < 0 else [0.0, _posD])
        ch_q = alt.Chart(df_q).mark_bar().encode(
            x=x_axis,
            y=alt.Y('Q:Q', scale=alt.Scale(domain=_dom_q)),
            color=alt.value('#d62728'),
            tooltip=['index:O', alt.Tooltip('Q:Q', format='.3f')]
        ).properties(width=st.session_state['current_chart_width'], height=st.session_state['current_chart_height'], title=alt.TitleParams('每动作 Q 值', anchor='middle', fontSize=22))
        cur_cols_num = int(st.session_state.get('current_cols', 3))
        if cur_cols_num >= 3:
            cur_cols = st.columns(3)
            cur_cols[0].altair_chart(ch_q.interactive(), use_container_width=False)
        elif cur_cols_num == 2:
            row1 = st.columns(2)
            row1[0].altair_chart(ch_q.interactive(), use_container_width=False)
        else:
            st.altair_chart(ch_q.interactive(), use_container_width=True)

        n_disp = len(ACTION_LABELS)
        idxs_map = list(range(n_disp))
        labels_map = [ACTION_LABELS[i] for i in idxs_map]
        indices_html = ''.join([f"<div style='padding:6px 0'>{i}</div>" for i in idxs_map])
        labels_html = ''.join([f"<div style='padding:6px 0;font-weight:600;color:#333'>{label}</div>" for label in labels_map])
        st.markdown(
            f"<div style='margin-top:24px;text-align:center'>"
            f"<div style='font-size:18px;font-weight:700;margin-bottom:10px'>动作索引与标签</div>"
            f"<div style='display:grid;grid-template-columns:repeat({n_disp}, minmax(0,1fr));gap:6px;justify-items:center;font-size:16px;color:#444'>{indices_html}</div>"
            f"<div style='display:grid;grid-template-columns:repeat({n_disp}, minmax(0,1fr));gap:6px;justify-items:center;font-size:14px'>{labels_html}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        

        # 最近动作组件已移除
    elif view_mode == '历史趋势':
        # 历史趋势：从 CSV 读取，支持选择动作与事件并显示折线图
        if not os.path.exists(CSV_PATH):
            placeholder_info.warning('未找到历史 CSV：正在等待训练写入 train_metrics.csv ...')
        else:
            try:
                df = pd.read_csv(CSV_PATH, on_bad_lines='skip')
                if len(df) == 0:
                    placeholder_info.warning('CSV 为空，等待写入...')
                else:
                    runs = df['run_id'].unique().tolist() if 'run_id' in df.columns else ['unknown']
                    sel_run = st.sidebar.selectbox('选择运行(run_id)', runs, index=max(0, len(runs)-1))
                    if 'run_id' in df.columns:
                        df = df[df['run_id'] == sel_run]
                    
                    # 获取当前最新步数
                    current_step = int(df['step'].max()) if len(df) > 0 else 0
                    
                    # 检查是否需要刷新历史趋势（基于步数或时间）
                    should_refresh_history = False
                    if history_refresh_interval is not None:
                        if history_refresh_mode == '每2000步':
                            last_step = int(st.session_state['last_history_step'])
                            step_diff = current_step - last_step
                            if step_diff >= 2000:
                                should_refresh_history = True
                                st.session_state['last_history_step'] = current_step
                        elif history_refresh_mode == '每3分钟':
                            should_refresh_history = True
                    
                    # 解析列表字符串
                    def parse_list(s):
                        try:
                            if pd.isna(s) or s == '' or s is None:
                                return []
                            return [float(x) for x in str(s).split(';') if x.strip() != '']
                        except Exception as e:
                            print(f"解析列表失败: {s}, 错误: {e}")
                            return []
                    
                    # 推断动作与事件数量
                    last_q = parse_list(df['q_values'].iloc[-1])
                    # last_des = parse_list(df['desire_thresholds'].iloc[-1])
                    # last_ev = parse_list(df['reward_thresholds_action'].iloc[-1])
                    num_actions = len(last_q) # max(len(last_q), len(last_des))
                    # num_events = len(last_ev)
                    
                    act_options = [f"{i} {ACTION_LABELS[i] if i < len(ACTION_LABELS) else ''}".strip() for i in range(num_actions)]
                    
                    # 获取当前选择的动作，优先使用会话状态中缓存的值
                    cached_action = st.session_state.get('cached_sel_action', 0)
                    default_index = min(cached_action, len(act_options)-1) if act_options else 0
                    
                    sel_act_label = st.sidebar.selectbox('选择动作', act_options, index=default_index)
                    sel_action = act_options.index(sel_act_label)
                    
                    # 检查是否需要重新计算数据（动作切换或强制刷新）
                    action_changed = st.session_state.get('cached_sel_action') != sel_action
                    _cur_auto = st.session_state.get('use_auto_line_ratio_hist', True)
                    _cur_ratio = float(st.session_state.get('history_line_fill_ratio', 0.66))
                    _cur_center_mode = st.session_state.get('center_mode_hist', '中点居中')
                    _prev_auto = st.session_state.get('prev_use_auto_line_ratio_hist')
                    _prev_ratio = st.session_state.get('prev_history_line_fill_ratio')
                    _prev_center_mode = st.session_state.get('prev_center_mode_hist')
                    config_changed = (_prev_auto is None or _prev_auto != _cur_auto) or (_prev_ratio is None or _prev_ratio != _cur_ratio) or (_prev_center_mode is None or _prev_center_mode != _cur_center_mode)
                    st.session_state['prev_use_auto_line_ratio_hist'] = _cur_auto
                    st.session_state['prev_history_line_fill_ratio'] = _cur_ratio
                    st.session_state['prev_center_mode_hist'] = _cur_center_mode
                    should_recalculate = should_refresh_history or action_changed or st.session_state.get('force_history_refresh', False) or config_changed
                    
                    # 更新会话状态中的缓存值
                    st.session_state['cached_sel_action'] = sel_action
                    
                    if should_recalculate:
                        # 清除图表缓存以确保重新生成
                        if action_changed or config_changed:
                            st.session_state.pop('cached_charts', None)
                        
                        # 只在需要时重新计算历史趋势
                        if should_refresh_history or action_changed or st.session_state.get('force_history_refresh', False):
                            st.session_state['force_history_refresh'] = False
                            
                            # 缓存解析后的数据以避免重复计算
                            if 'cached_df' not in st.session_state or action_changed:
                                steps = df['step'].tolist()
                                
                                # 解析Q值
                                q_series = []
                                for s in df['q_values']:
                                    q_list = parse_list(s)
                                    q_val = q_list[sel_action] if len(q_list) > sel_action else None
                                    q_series.append(q_val)
                            
                                # 缓存数据
                                st.session_state['cached_df'] = {
                                    'steps': steps,
                                    'q_series': q_series
                                }
                        
                        # 使用缓存的数据
                        cached_data = st.session_state['cached_df']
                        df_hist = pd.DataFrame({
                            'step': cached_data['steps'], 
                            'Q': cached_data['q_series']
                        })
                        
                        # 过滤掉None值
                        df_hist = df_hist.dropna()
                        
                        if len(df_hist) > 0:
                            _mx_hist = df_hist['Q'].max()
                            _mn_hist = df_hist['Q'].min()
                            _dom_hist = [hist_y_min, hist_y_max]
                            if st.session_state.get('use_auto_line_ratio_hist', True):
                                _fill_hist = float(st.session_state.get('history_line_fill_ratio', 0.66))
                                _center_mode = st.session_state.get('center_mode_hist', '中点居中')
                                if _center_mode == '零居中':
                                    _M = (max(abs(_mx_hist), abs(_mn_hist)) / _fill_hist) if (_mx_hist != 0 or _mn_hist != 0) else 1.0
                                    _dom_hist = [-_M, _M]
                                elif _center_mode == '中值居中':
                                    _med = float(df_hist['Q'].median()) if len(df_hist) > 0 else 0.0
                                    _up = (_mx_hist - _med) / _fill_hist if _mx_hist != _med else 1.0
                                    _lo = (_med - _mn_hist) / _fill_hist if _mn_hist != _med else 1.0
                                    _span = max(_up, _lo, 1.0)
                                    _dom_hist = [_med - _span, _med + _span]
                                elif _center_mode == '中点居中':
                                    _mid = (_mx_hist + _mn_hist) / 2.0
                                    _up = (_mx_hist - _mid) / _fill_hist if _mx_hist != _mid else 1.0
                                    _lo = (_mid - _mn_hist) / _fill_hist if _mn_hist != _mid else 1.0
                                    _span = max(_up, _lo, 1.0)
                                    _dom_hist = [_mid - _span, _mid + _span]
                                else:
                                    _posH = (_mx_hist / _fill_hist) if _mx_hist > 0 else 1.0
                                    _negH = (abs(_mn_hist) / _fill_hist) if _mn_hist < 0 else 0.0
                                    _dom_hist = ([-_negH, _posH] if _mn_hist < 0 else [0.0, _posH])
                            ch_hist = alt.Chart(df_hist).mark_line().encode(
                                    x='step:Q',
                                    y=alt.Y('Q:Q', scale=alt.Scale(domain=_dom_hist, nice=False)),
                                    tooltip=['step:Q', alt.Tooltip('Q:Q', format='.3f')]
                                ) \
                                .properties(width=700, height=450)
                        else:
                            ch_hist = None
                        
                        # 创建事件阈值图表
                        if len(cached_data['rows_ev']) > 0:
                            df_ev_hist = pd.DataFrame(cached_data['rows_ev'])
                            
                            # 调试信息：显示数据样本
                            if len(df_ev_hist) > 0:
                                print(f"事件阈值数据样本: {df_ev_hist.head()}")
                                print(f"事件标签: {df_ev_hist['event_label'].unique()}")
                                print(f"步数范围: {df_ev_hist['step'].min()} - {df_ev_hist['step'].max()}")
                                print(f"阈值范围: {df_ev_hist['RewardThreshold'].min()} - {df_ev_hist['RewardThreshold'].max()}")
                            
                            _mx_ev_hist = (df_ev_hist['RewardThreshold'].max() if len(df_ev_hist) > 0 else 0.0)
                            _mn_ev_hist = (df_ev_hist['RewardThreshold'].min() if len(df_ev_hist) > 0 else 0.0)
                            _dom_ev_hist = [event_hist_y_min, event_hist_y_max]
                            if st.session_state.get('use_auto_line_ratio_hist', True):
                                _fill_hist = float(st.session_state.get('history_line_fill_ratio', 0.66))
                                _center_mode = st.session_state.get('center_mode_hist', '中点居中')
                                if _center_mode == '零居中':
                                    _M2 = (max(abs(_mx_ev_hist), abs(_mn_ev_hist)) / _fill_hist) if (_mx_ev_hist != 0 or _mn_ev_hist != 0) else 1.0
                                    _dom_ev_hist = [-_M2, _M2]
                                elif _center_mode == '中值居中':
                                    _med2 = float(df_ev_hist['RewardThreshold'].median()) if len(df_ev_hist) > 0 else 0.0
                                    _up2 = (_mx_ev_hist - _med2) / _fill_hist if _mx_ev_hist != _med2 else 1.0
                                    _lo2 = (_med2 - _mn_ev_hist) / _fill_hist if _mn_ev_hist != _med2 else 1.0
                                    _span2 = max(_up2, _lo2, 1.0)
                                    _dom_ev_hist = [_med2 - _span2, _med2 + _span2]
                                elif _center_mode == '中点居中':
                                    _mid2 = (_mx_ev_hist + _mn_ev_hist) / 2.0
                                    _up2 = (_mx_ev_hist - _mid2) / _fill_hist if _mx_ev_hist != _mid2 else 1.0
                                    _lo2 = (_mid2 - _mn_ev_hist) / _fill_hist if _mn_ev_hist != _mid2 else 1.0
                                    _span2 = max(_up2, _lo2, 1.0)
                                    _dom_ev_hist = [_mid2 - _span2, _mid2 + _span2]
                                else:
                                    _posE2 = (_mx_ev_hist / _fill_hist) if _mx_ev_hist > 0 else 1.0
                                    _negE2 = (abs(_mn_ev_hist) / _fill_hist) if _mn_ev_hist < 0 else 0.0
                                    _dom_ev_hist = ([-_negE2, _posE2] if _mn_ev_hist < 0 else [0.0, _posE2])
                            ch_ev_hist = alt.Chart(df_ev_hist).mark_line().encode(
                                x='step:Q',
                                y=alt.Y('RewardThreshold:Q', scale=alt.Scale(domain=_dom_ev_hist, nice=False)),
                                color=alt.Color('event_label:N', legend=alt.Legend(orient='right'))
                            ).properties(width=700, height=450)
                        else:
                            ch_ev_hist = None
                            print("警告：没有事件阈值数据可用")
                        
                        # 缓存图表
                        st.session_state['cached_charts'] = {
                            'ch_hist': ch_hist,
                            'ch_ev_hist': ch_ev_hist
                        }
                    
                    # 显示图表（使用缓存或新计算的结果）
                    charts = st.session_state.get('cached_charts', {})
                    
                    # 获取当前选择的动作索引（已从会话状态中更新）
                    sel_action = st.session_state.get('cached_sel_action', 0)
                    
                    # 第一行：奖励历史趋势 + 奖励均值历史
                    try:
                        df_rewards = pd.DataFrame({
                            'step': df['step'].tolist(),
                            'AdjReward': df['adj_reward'].tolist(),
                            'RawReward': df['raw_reward'].tolist() if 'raw_reward' in df.columns else [None]*len(df['step'].tolist())
                        })
                        df_rewards = df_rewards.dropna()
                        _mx_r = max(df_rewards['AdjReward'].max(), df_rewards['RawReward'].max() if 'RawReward' in df_rewards.columns and len(df_rewards['RawReward'].dropna())>0 else 0.0)
                        _mn_r = min(df_rewards['AdjReward'].min(), df_rewards['RawReward'].min() if 'RawReward' in df_rewards.columns and len(df_rewards['RawReward'].dropna())>0 else 0.0)
                        _dom_r = [hist_y_min, hist_y_max]
                        if st.session_state.get('use_auto_line_ratio_hist', True):
                            _fill_hist = float(st.session_state.get('history_line_fill_ratio', 0.66))
                            _posR = (_mx_r / _fill_hist) if _mx_r > 0 else 1.0
                            _negR = (abs(_mn_r) / _fill_hist) if _mn_r < 0 else 0.0
                            _dom_r = ([-_negR, _posR] if _mn_r < 0 else [0.0, _posR])
                        def _ev_to_labels(s):
                            try:
                                parts = [int(x) for x in str(s).split(';') if str(x).strip() != '']
                                labels = [EVENT_LABELS[i] if i < len(EVENT_LABELS) else str(i) for i in parts]
                                return '|'.join(labels)
                            except Exception:
                                return ''
                        def _evfb_to_str(s):
                            try:
                                itms = [p for p in str(s).split('|') if p]
                                pairs = []
                                for it in itms:
                                    kv = it.split(':')
                                    if len(kv) == 2:
                                        ei = int(kv[0])
                                        fb = float(kv[1])
                                        label = EVENT_LABELS[ei] if ei < len(EVENT_LABELS) else str(ei)
                                        pairs.append(f"{label}:{fb:.3f}")
                                return '|'.join(pairs)
                            except Exception:
                                return ''
                        df_rewards['events_str'] = [ _ev_to_labels(s) for s in (df['events'].tolist() if 'events' in df.columns else ['']*len(df['step'].tolist())) ]
                        df_rewards['events_feedback_str'] = [ _evfb_to_str(s) for s in (df['events_feedback'].tolist() if 'events_feedback' in df.columns else ['']*len(df['step'].tolist())) ]
                        ch_rewards = alt.Chart(df_rewards).transform_fold(['AdjReward', 'RawReward'], as_=['type', 'value']) \
                            .mark_line().encode(
                                x='step:Q',
                                y=alt.Y('value:Q', scale=alt.Scale(domain=_dom_r, nice=False)),
                                color=alt.Color('type:N', legend=alt.Legend(orient='bottom', title=None)),
                                tooltip=['step:Q', 'type:N', alt.Tooltip('value:Q', format='.3f'), 'events_str:N', 'events_feedback_str:N']
                            ).properties(width=700, height=450, title=alt.TitleParams('奖励 历史趋势', anchor='middle', fontSize=22))

                        df_mean = pd.DataFrame({
                            'step': df['step'].tolist(),
                            'RewardAvgRecent': df['reward_avg_recent'].tolist() if 'reward_avg_recent' in df.columns else [None]*len(df['step'].tolist())
                        })
                        df_mean = df_mean.dropna()
                        _mx_m = (df_mean['RewardAvgRecent'].max() if len(df_mean)>0 else 0.0)
                        _mn_m = (df_mean['RewardAvgRecent'].min() if len(df_mean)>0 else 0.0)
                        _dom_m = [hist_y_min, hist_y_max]
                        if st.session_state.get('use_auto_line_ratio_hist', True):
                            _fill_hist = float(st.session_state.get('history_line_fill_ratio', 0.66))
                            _posM = (_mx_m / _fill_hist) if _mx_m > 0 else 1.0
                            _negM = (abs(_mn_m) / _fill_hist) if _mn_m < 0 else 0.0
                            _dom_m = ([-_negM, _posM] if _mn_m < 0 else [0.0, _posM])
                        ch_mean = alt.Chart(df_mean).mark_line(color='#1f77b4').encode(
                            x='step:Q',
                            y=alt.Y('RewardAvgRecent:Q', scale=alt.Scale(domain=_dom_m, nice=False)),
                            tooltip=['step:Q', alt.Tooltip('RewardAvgRecent:Q', format='.3f')]
                        ).properties(width=700, height=450, title=alt.TitleParams('奖励均值 历史趋势', anchor='middle', fontSize=22))

                        row_rewards = st.columns(2)
                        row_rewards[0].altair_chart(ch_rewards.interactive(), use_container_width=False)
                        row_rewards[1].altair_chart(ch_mean.interactive(), use_container_width=False)
                    except Exception:
                        pass

                    # 第二行：只显示 Q 值历史趋势
                    row_top = st.columns(2)
                    row_top[0].subheader(f'动作 {sel_action} 的 Q 值 历史趋势')
                    if charts.get('ch_hist') is not None:
                        row_top[0].altair_chart(charts['ch_hist'].interactive(), use_container_width=False)
                    else:
                        row_top[0].warning('暂无数据')
                    
                    # 只在需要时重新计算所有动作的 Q 值图表
                    if should_refresh_history or action_changed or config_changed or 'cached_all_charts' not in st.session_state:
                        rows_q = []
                        for idx, r in df.iterrows():
                            q_list = parse_list(r['q_values'])
                            for ai in range(len(q_list)):
                                rows_q.append({'step': r['step'], 'action': ai, 'value': q_list[ai]})
                        
                        df_all_q = pd.DataFrame(rows_q)
                        
                        _mx_all_q = (df_all_q['value'].max() if len(df_all_q) > 0 else 0.0)
                        _mn_all_q = (df_all_q['value'].min() if len(df_all_q) > 0 else 0.0)
                        _dom_all_q = [hist_y_min, hist_y_max]
                        if st.session_state.get('use_auto_line_ratio_hist', True):
                            _fill_hist = float(st.session_state.get('history_line_fill_ratio', 0.66))
                            _center_mode = st.session_state.get('center_mode_hist', '中点居中')
                            if _center_mode == '零居中':
                                _M3 = (max(abs(_mx_all_q), abs(_mn_all_q)) / _fill_hist) if (_mx_all_q != 0 or _mn_all_q != 0) else 1.0
                                _dom_all_q = [-_M3, _M3]
                            elif _center_mode == '中值居中':
                                _med3 = float(df_all_q['value'].median()) if len(df_all_q) > 0 else 0.0
                                _up3 = (_mx_all_q - _med3) / _fill_hist if _mx_all_q != _med3 else 1.0
                                _lo3 = (_med3 - _mn_all_q) / _fill_hist if _mn_all_q != _med3 else 1.0
                                _span3 = max(_up3, _lo3, 1.0)
                                _dom_all_q = [_med3 - _span3, _med3 + _span3]
                            elif _center_mode == '中点居中':
                                _mid3 = (_mx_all_q + _mn_all_q) / 2.0
                                _up3 = (_mx_all_q - _mid3) / _fill_hist if _mx_all_q != _mid3 else 1.0
                                _lo3 = (_mid3 - _mn_all_q) / _fill_hist if _mn_all_q != _mid3 else 1.0
                                _span3 = max(_up3, _lo3, 1.0)
                                _dom_all_q = [_mid3 - _span3, _mid3 + _span3]
                            else:
                                _posAQ = (_mx_all_q / _fill_hist) if _mx_all_q > 0 else 1.0
                                _negAQ = (abs(_mn_all_q) / _fill_hist) if _mn_all_q < 0 else 0.0
                                _dom_all_q = ([-_negAQ, _posAQ] if _mn_all_q < 0 else [0.0, _posAQ])
                        ch_all_q = alt.Chart(df_all_q).mark_line().encode(
                            x='step:Q',
                            y=alt.Y('value:Q', title='Q', scale=alt.Scale(domain=_dom_all_q, nice=False)),
                            color=alt.Color('action:N', legend=alt.Legend(orient='bottom', title='索引'))
                        ).properties(width=700, height=450)
                        
                        st.session_state['cached_all_charts'] = {
                            'ch_all_q': ch_all_q
                        }
                    
                    all_charts = st.session_state.get('cached_all_charts', {})
                    row3 = st.columns(2)
                    row3[0].subheader('所有动作 Q 值 历史趋势')
                    if all_charts.get('ch_all_q') is not None:
                        row3[0].altair_chart(all_charts['ch_all_q'].interactive(), use_container_width=False)
                    
                    # 添加手动刷新按钮
                    if history_refresh_mode == '手动':
                        if st.sidebar.button('手动刷新历史趋势'):
                            st.session_state['force_history_refresh'] = True
                            st.rerun()
                    
            except Exception as e:
                placeholder_info.error(f'读取 CSV 失败：{e}')
                import traceback
                traceback.print_exc()

# 将Y轴范围设置放在侧边栏底部，并提供复位按钮
# 图尺寸与布局设置
if view_mode == '当前步':
    st.sidebar.subheader('柱高度占比')
    st.sidebar.checkbox('启用自适应柱高', value=st.session_state.get('use_auto_bar_ratio', True), key='use_auto_bar_ratio')
    st.sidebar.slider('柱高度占比', 0.30, 0.95, value=st.session_state.get('current_bar_fill_ratio', 0.66), key='current_bar_fill_ratio')
elif view_mode == '历史趋势':
    st.sidebar.subheader('图高度占比')
    st.sidebar.checkbox('启用自适应图高', value=st.session_state.get('use_auto_line_ratio_hist', True), key='use_auto_line_ratio_hist')
    st.sidebar.slider('图高度占比', 0.30, 0.95, value=st.session_state.get('history_line_fill_ratio', 0.66), key='history_line_fill_ratio')
    st.sidebar.selectbox('居中模式', ['不居中', '零居中', '中值居中', '中点居中'], index=3, key='center_mode_hist')

# 自动刷新降级方案：当未安装 streamlit-autorefresh 插件时，使用 sleep + rerun
if (not _HAS_AR) and st.session_state.get('enable_auto_checkbox', False):
    try:
        time.sleep(float(autorefresh))
        st.rerun()
    except Exception:
        pass