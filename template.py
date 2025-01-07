import streamlit as st


st.set_page_config(page_title="國立虎尾科技大學機械設計工程系", layout="wide")


st.title("113(上) 智慧機械設計 課程期末報告")

#st.title("ROS自主移動平台與AI整合之研究")

st.markdown("## <span style='color:red;'>ROS自主移動平台與AI整合之研究</span>", unsafe_allow_html=True)
st.divider()
st.markdown("### 指導老師：周榮源 教授")
st.markdown("### 班級：碩設計一甲")
st.markdown("### 組別：第 5 組")

st.markdown("#### 組員：")
st.markdown("#### 41023146 洪偉陞")
st.markdown("#### 11373107 謝帆俊")
st.markdown("#### 11373102 詹宗樺")
st.divider()
st.markdown("##### 網頁網址：https://computervision-mk5fwf8sxraff5a5iffgaz.streamlit.app/ ")
st.markdown("##### 倉儲網址：https://github.com/41023146/computer_vision ")
st.divider()
st.write("歡迎來到智慧機械設計課程報告。請從左側選單選擇要查看的項目。")

# 不需要在這裡定義實驗項目列表，因為Streamlit會自動生成側邊欄

# 顯示一張圖片(image)
st.image("files/NFU.png")
# Caption
st.caption("""(這是國立虎尾科技大學之校門!)""")

