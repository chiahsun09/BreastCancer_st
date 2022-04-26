

import time
import streamlit as st
import numpy as np
import pandas as pd


# 網頁配置設定(要寫在所有 Streamlit 命令之前，而且只能設定一次)
st.set_page_config(
    page_title="自定義網頁標題",
    page_icon="random",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# 加入標題
st.title('streamlit測試')


# 使用 Magic commands 指令，顯示 Markdown
st.write("嘗試創建**表格**：")

df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})
# 單行只有變數，不需要使用 st.write()，它會自動套用
df


# 繪製折線圖
# 使用 Numpy 生成一個隨機樣本，然後將其繪製成圖表。
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])
st.line_chart(chart_data)