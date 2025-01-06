import streamlit as st

st.title("第四章 結論與心得")

st.write("這裡是流體靜力學實驗的詳細內容。")



tab1, tab2 = st.tabs(["4.1","4.2"])
# 在這裡添加實驗一的具體內容，如圖表、數據等
with tab1:
    st.header("4.1結論")
    st.write("將此實驗之目的以文字結合圖片之方式描述於此!")

with tab2:
    st.header("4.2心得")
    st.write("將此實驗之目的以文字結合圖片之方式描述於此!")

