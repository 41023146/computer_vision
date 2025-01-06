import streamlit as st

st.title("第三章 結果與討論")

st.write("以下為各部分的結果")



tab1, tab2, tab3, tab4, tab5 = st.tabs(["3.1","3.2","3.3","3.4","3.5"])
# 在這裡添加實驗一的具體內容，如圖表、數據等
with tab1:
    st.header("3.1Turtlebot3(Burger)避障與導航測試結果")
    st.markdown("##### 測試中使用鍵盤控制機器人移動同時建構地圖")
    sample_video = open("files/3_1_01.mp4", "rb").read()
    # Display Video using st.video() function
    st.video(sample_video, start_time = 0)
    st.divider()
    st.markdown("##### 機器人的避障及導航")
    sample_video = open("files/3_1_02.mp4", "rb").read()
    # Display Video using st.video() function
    st.video(sample_video, start_time = 0)
    st.divider()
    

with tab2:
    st.header("3.2 ROS/AMR之深度學習影像移動控制結果")
    st.markdown("##### 訓練完成可辨識方向鍵號之結果影片")
    sample_video = open("files/3_2_01.mp4", "rb").read()
    # Display Video using st.video() function
    st.video(sample_video, start_time = 0)
    st.markdown("##### 1.右轉")
    st.image("files/3_2_01.png")
    st.markdown("##### 2.左轉")
    st.image("files/3_2_02.png")
    st.markdown("##### 3.迴轉")
    st.image("files/3_2_03.png")
    st.markdown("##### 4.前進")
    st.image("files/3_2_04.png")
    

with tab3:
    st.header("3.3 ultralytics YOLO與SAM物件偵測與分割結果")
    st.markdown("##### 以下是YOLOYOLO與SAM物件偵測與分割之結果影片")
    sample_video = open("files/3_3_01.mp4", "rb").read()
    # Display Video using st.video() function
    st.video(sample_video, start_time = 0)
    st.divider()
    st.markdown("##### 影片中偵測與分割的三個物件：")
    
    st.markdown("##### 1.杏鮑菇")
    st.image("files/3_3_02.png")
    st.markdown("##### 2.蘋果")
    st.image("files/3_3_01.png")
    st.markdown("##### 3.香蕉")
    st.image("files/3_3_03.png")
    
    
with tab4:
    st.header("3.4 three-link planar manipulator模擬結果")
    st.markdown("##### env.py程式執行，會可視化三連趕機械臂隨機動作的情況")
    sample_video = open("files/3_4_01.mp4", "rb").read()
    # Display Video using st.video() function
    st.video(sample_video, start_time = 0)
    st.divider()
    st.markdown("##### 在ON_TRAIN = True時，執行main.py程式時會訓練模型")
    sample_video = open("files/3_4_02.mp4", "rb").read()
    # Display Video using st.video() function
    st.video(sample_video, start_time = 0)
    st.divider()
    st.markdown("##### 訓練完模型會存成下圖的四個檔案")
    st.image("files/3_4_01.png")
    st.divider()
    st.markdown("##### ON_TRAIN = False時，執行main.py程式時會評估模型，並使用env.py可視化")
    sample_video = open("files/3_4_03.mp4", "rb").read()
    # Display Video using st.video() function
    st.video(sample_video, start_time = 0)
    st.divider()
with tab5:
    st.header("3.5 streamlit UI設計與資料可視化結果")
    st.markdown("##### 以下為streamlit網頁格式的期末報告")

    sample_video = open("files/3_5_01.mp4", "rb").read()
    # Display Video using st.video() function
    st.video(sample_video, start_time = 0)
    
    st.markdown("##### 網業網址：https://computervision-mk5fwf8sxraff5a5iffgaz.streamlit.app/ ")






