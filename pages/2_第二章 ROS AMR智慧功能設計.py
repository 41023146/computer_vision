import streamlit as st

st.title("第二章 ROS AMR智慧功能設計")

st.write("以AI與AMR整合功能之設計與應用為範疇，詳細說明下列子項之研究方法與步驟：")



tab1, tab2, tab3, tab4, tab5 = st.tabs(["2.1","2.2","2.3","2.4","2.5"])
# 在這裡添加實驗一的具體內容，如圖表、數據等
with tab1:
    st.header("2.1 Turtlebot3(Burger)之避障與導航實作")
    st.write("首先建立名為catkin_ws資料夾")
    st.image("files/2_1_01.png")
    st.write("然後在catkin_ws中新增src資料夾")
    st.divider()
    st.write("輸入:git clone https://github.com/ROBOTIS-GIT/turtlebot3.git ，將turtlebot3的專案clone到近端(終端機1)")
    st.write("""
    安裝以下套件：(終端機2)\n
    sudo apt install ros-noetic-turtlebot3\n
    sudo apt install ros-noetic-gmapping\n
    sudo apt install ros-noetic-hector-slam\n
    sudo apt install ros-noetic-slam-gmapping\n
    sudo apt install ros-noetic-rviz\n
    sudo apt-get install ros-noetic-dwa-local-planner\n
    sudo apt install ros-noetic-turtlebot3-msgs\n
    sudo apt install ros-noetic-turtlebot3-navigation\n
    sudo apt install ros-noetic-turtlebot3-teleop\n
    """)
    st.divider()
    st.write("執行catkin_make編譯工作空間(終端機1)")
    st.image("files/2_1_02.png")
    st.divider()
    st.write("執行source devel/setup.bash設定環境變數(終端機1)")
    st.image("files/2_1_06.png")
    st.divider()
    st.write("執行ping 192.168.50.14確認可以互相ping(終端機1)")
    st.image("files/2_1_03.png")
    st.divider()
    st.write("啟動roscore(終端機3)")
    st.image("files/2_1_04.png")
    
    st.divider()
    st.write("輸入ssh pi@192.168.1.199，連線。密碼為: raspberry(終端機4)")
    st.image("files/2_1_05.png")
    st.divider()
    
    st.write("輸入: roslaunch turtlebot3_bringup turtlebot3_robot.launch，啟動turtlebot3的硬體節點，使機器人的感測器跟行動裝置可以跟ROS通訊(終端機4)")
    st.image("files/2_1_07.png")
    st.divider()
    st.write("輸入: export TURTLEBOT3_MODEL=burger ，確定機器人型號為burger(終端機5)")
    st.image("files/2_1_08.png")
    st.divider()
    st.write("輸入:roslaunch turtlebot3_slam turtlebot3_slam.launch，啟動Slam建構地圖(終端機6)")
    st.image("files/2_1_09.png")
    st.divider()
    st.write("輸入:roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch，啟動鍵盤操控(終端機6)")
    st.image("files/2_1_10.png")
    st.divider()
    st.write("輸入:rosrun map_server map_saver -f ~/map，儲存建構完的地圖(終端機7)")
    st.image("files/2_1_11.png")
    st.divider()
    st.write("輸入:roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=$HOME/map.yaml，啟動導航(終端機7)")
    st.image("files/2_1_12.png")
    st.divider()
    
    st.write("輸入:roslaunch turtlebot3_navigation view_navigation.launch，開啟Rviz設定方向及目標(終端機8)")
    st.image("files/2_1_13.png")
    st.image("files/2_1_14.png",caption="方向")
    st.image("files/2_1_15.png",caption="目標")
    st.divider()
    




with tab2:
    st.header("2.2 ROS/AMR之深度學習影像移動控制功能實作")
    st.markdown("##### 下載標註用程式：")
    code = '''
    git clone https://github.com/HumanSignal/labelImg.git
    pip install PyQt5
    pip install lxml
    pyrcc5 -o libs/resources.py resources.qrc
    '''
    st.code(code, language='python')
    st.divider()
    st.write("1.對資料集標註")
    st.image("files/2_2_01.png")
    st.divider()
    st.write("2.建立yaml")
    code = '''
    names:
    - '0': left
    - '1': right
    - '2': turn around
    - '3': go straight
    - '4': stop
    test: dataset/test
    train: dataset/train
    val: dataset/valid
    '''
    st.code(code, language='python')
    st.divider()
    st.write("3.訓練模型")
    st.image("files/2_2_02.png")
    st.image("files/2_2_03.png")
    st.divider()
    st.write("4.使用模型")
    st.image("files/2_2_04.png")
    st.image("files/2_2_05.png")
    


    
with tab3:
    st.header("2.3 ultralytics YOLO與SAM物件偵測與分割功能實作")
    st.write("SAM模型，參考自:https://github.com/noorkhokhar99/Segment-Anything-with-Webcam-in-Real-Time-with-FastSAM.git")
    st.divider()
    st.write("以下為程式及註解內容：")
    st.write("1.導入模組及設定變數")
    st.image("files/2_3_01.png")
    st.divider()
    st.write("2.處理SAM遮罩影像副函式 fast_show_mask_gpu()")
    st.image("files/2_3_02.png")
    st.divider()
    st.write("3.建立類別名稱")
    st.image("files/2_3_03.png")
    st.divider()
    st.write("4.模型、攝影機、顯示窗設定")
    st.image("files/2_3_04.png")
    st.divider()
    st.write("5.主迴圈cap.isOpened()")
    st.image("files/2_3_05.png")

with tab4:
    st.header("2.4 three-link planar manipulator模擬實作")
    
    st.write("這個題目主要目的是利用Reinforcement Learning(RL),訓練三桿件的平面機械手臂，使手臂依照訓練出來的權重判斷方塊位置(滑鼠控制)，要使用哪三個角度才可以使finger接觸到方塊")
    st.write("本專案參考 https://github.com/MorvanZhou/train-robot-arm-from-scratch 這個兩桿件平面機械手臂做修改")
    st.divider()
    
    st.write("首先介紹一下three-link planar manipulator這個專案所使用的環境：Python3.10.11、gym 0.15.4、tensorflow 2.18.0")
    st.image("files/2_4_01.png")
    st.divider()
    
    st.write("這個專案主要有三個程式分別是：main.py、env.py、rl.py")
    st.write("主程式main.py主要控制是要訓練模型還是使用模型，env.py是用來建立機械手臂的環境，並且手臂可視化也在這個部分，而最後的rl.py則是強化學習的程式，三個程式放在同一目錄下，在main.py中以下方的程式導入。")
    code = '''
    from env import ArmEnv
    from rl import DDPG
    '''
    st.code(code, language='python')
    st.divider()
    
    st.write("再來先介紹一下env.py這個程式")
    st.write("首先因為參考的程式太過老舊需要先處理版本差異所導致的問題，在env.py程式中需用到pyglet來做可視化但pyglet的版本為1.3.2，其中會使用到time.clock，但在python3.8之後已經移除了time.clock，所以我用以下的方式將time.clock替換掉")
    code = '''
    # 替代 time.clock，讓舊版 pyglet 兼容 Python 3.8+
    if not hasattr(time, 'clock'):
        time.clock = time.perf_counter
    '''
    st.code(code, language='python')
    st.divider()
    st.write("1.模組的導入")
    code = '''
    import numpy as np
    import pyglet
    import time
    '''
    st.code(code, language='python')
    st.write("""
    引入所需模組：\n
    numpy：進行數學運算與矩陣操作。\n
    pyglet：提供圖形界面，用於可視化模擬。
    """)
    st.divider()
    st.write("2.主類別(class) ArmEnv")
    code = '''
    class ArmEnv(object):
        viewer = None
        dt = .1    # 刷新頻率
        action_bound = [-1, 1]
        goal = {'x': 100., 'y': 100., 'l': 40}
        state_dim = 13  # 狀態空間維度
        action_dim = 3  # 動作空間維度（三段機械臂）
    

    '''
    st.code(code, language='python')
    st.write("""
    類別的屬性：\n       
    viewer：渲染器物件。\n 
    dt：每次刷新時間間隔。\n 
    action_bound：動作（角度變化）的範圍 [−1,1]。\n 
    goal：目標點的資訊（x、y 表示位置，l 表示目標框的大小）。\n 
    state_dim 和 action_dim：分別表示狀態空間和動作空間的維度。  
    """)
    st.divider()
    
    st.write("3.init初始化")
    code = '''
    def __init__(self):
        self.arm_info = np.zeros(
            3, dtype=[('l', np.float32), ('r', np.float32)])
        total_length = 200
        self.arm_info['l'] = total_length / 3
        self.arm_info['r'] = np.pi / 6
        self.on_goal = 0
    '''
    st.code(code, language='python')
    st.write("""    
    初始化機械臂環境，包括臂的長度、角度，以及其他狀態變數。\n 
    self.arm_info是使用 NumPy 結構型陣列儲存每段機械臂的長度（l）和角度（r）。\n 
    total_length是機械臂的總長度，將其固定為 200，均分為 3 段​。\n 
    self.on_goal會記錄機械臂是否穩定停留在目標上，初始值為 0。
    """)    
    st.divider()
    
    st.write("4.step步驟")
    code = '''
    def step(self, action):
        done = False
        action = np.clip(action, *self.action_bound)
        self.arm_info['r'] += action * self.dt
        self.arm_info['r'] %= np.pi * 2  # normalize

        # 計算每段機械臂的終點位置
        (a1l, a2l, a3l) = self.arm_info['l']  # 每段臂長
        (a1r, a2r, a3r) = self.arm_info['r']  # 每段角度
        a1xy = np.array([200., 200.])  # 第一段起點
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # 第一段終點
        a2xy_ = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # 第二段終點
        finger = np.array([np.cos(a1r + a2r + a3r), np.sin(a1r + a2r + a3r)]) * a3l + a2xy_  # 第三段終點

        # 距離計算（歸一化）
        dist1 = [(self.goal['x'] - a1xy_[0]) / 400, (self.goal['y'] - a1xy_[1]) / 400]
        dist2 = [(self.goal['x'] - a2xy_[0]) / 400, (self.goal['y'] - a2xy_[1]) / 400]
        dist3 = [(self.goal['x'] - finger[0]) / 400, (self.goal['y'] - finger[1]) / 400]
        r = -np.sqrt(dist3[0]**2 + dist3[1]**2)

        # 判斷是否達到目標
        if (self.goal['x'] - self.goal['l']/2 < finger[0] < self.goal['x'] + self.goal['l']/2
        ) and (self.goal['y'] - self.goal['l']/2 < finger[1] < self.goal['y'] + self.goal['l']/2):
            r += 1.
            self.on_goal += 1
            if self.on_goal > 50: #要待一陣子才會變True
                done = True
        else:
            self.on_goal = 0

        # 狀態空間
        #串接數據
        s = np.concatenate((
            a1xy_/200, a2xy_/200, finger/200,  # 三段終點位置
            dist1 + dist2 + dist3,  # 與目標的相對距離
            [1. if self.on_goal else 0.]  # 是否在目標
        ))
        return s, r, done
    '''
    st.code(code, language='python')
    st.write("""
    Step根據輸入動作 action 更新機械臂的狀態，計算新狀態（s）、回報（r），並判斷回合是否結束（done）。\n
    限制動作的範圍（action_bound）。\n
    Action會隨機給三個值從-0.5到0.5，用來更新手臂的角度，並用餘數%=讓值不會超過2pi。\n
    依照長度跟角度透過三角函數計算出各個桿件末端的位置及距目標物的距離。\n
    若末端進入目標範圍並穩定超過 50 步，設定 done = True。\n
    創建狀態空間：包括機械臂的末端位置、距離目標的相對距離，以及是否在目標上的標誌。
     
    """)
    st.divider()
    st.write("5.reset重置")
    code = '''
    def reset(self):
        self.goal['x'] = np.random.rand()*400.
        self.goal['y'] = np.random.rand()*400.
        self.arm_info['r'] = 2 * np.pi * np.random.rand(3)  # 隨機初始化角度
        self.on_goal = 0
        (a1l, a2l, a3l) = self.arm_info['l']
        (a1r, a2r, a3r) = self.arm_info['r']
        a1xy = np.array([200., 200.])
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy
        a2xy_ = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_
        finger = np.array([np.cos(a1r + a2r + a3r), np.sin(a1r + a2r + a3r)]) * a3l + a2xy_

        dist1 = [(self.goal['x'] - a1xy_[0]) / 400, (self.goal['y'] - a1xy_[1]) / 400]
        dist2 = [(self.goal['x'] - a2xy_[0]) / 400, (self.goal['y'] - a2xy_[1]) / 400]
        dist3 = [(self.goal['x'] - finger[0]) / 400, (self.goal['y'] - finger[1]) / 400]

        s = np.concatenate((
            a1xy_/200, a2xy_/200, finger/200,
            dist1 + dist2 + dist3,
            [1. if self.on_goal else 0.]
        ))
        return s
    '''
    st.code(code, language='python')
    st.write("reset重置環境，在手臂末端碰到目標並返回done時會執行，隨機生成目標的位置 (x,y) ，範圍為 [0,400]，並隨機生成三段機械臂的初始角度。")
    st.divider()
    
    st.write("6.Viewer可視化")
    st.write("類別(class) Viewer主要就是利用arm_info跟goal兩個參數並以pyglet繪製，實現平面機械手臂的可視化。")
    st.write("可視化的視窗如下：")
    st.image("files/2_4_03.png")
    st.write("並且可以依據以下程式讓目標跟著滑鼠移動")
    code = '''
    def on_mouse_motion(self, x, y, dx, dy):
        self.goal_info['x'] = x
        self.goal_info['y'] = y

    '''
    st.code(code, language='python')
    
    st.write("rl.py 定義了 DDPG（Deep Deterministic Policy Gradient）強化學習演算法的核心邏輯，包括 Actor 和 Critic 網路的構建與更新、目標網路的軟更新、經驗記憶庫的管理、動作選擇，以及模型保存與加載（save() 和 restore()） ")
    st.write("作為智能體（Agent）的實現，rl.py 負責與 env.py 提供的模擬環境交互，通過 Actor 網路輸出動作並傳遞給環境的 step(action) 方法，獲取新狀態、回報及是否結束的信息")
    st.write("學習最優策略以最大化累計回報。")
    st.write("由於增強學習rl是直接使用同樣的框架，所以只需修改版本差異導致的錯誤即可")
    code = '''
    import tensorflow as tf
    import numpy as np

    #####################  hyper parameters  ####################

    LR_A = 0.001    # learning rate for actor
    LR_C = 0.001    # learning rate for critic
    GAMMA = 0.9     # reward discount
    TAU = 0.01      # soft replacement
    MEMORY_CAPACITY = 30000
    BATCH_SIZE = 32


    class DDPG:
        def __init__(self, a_dim, s_dim, a_bound):
            self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
            self.pointer = 0
            self.memory_full = False

            self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound[1]

            # Build actor and critic networks
            self.actor_eval = self._build_actor()
            self.actor_target = self._build_actor()
            self.critic_eval = self._build_critic()
            self.critic_target = self._build_critic()

            # Optimizers
            self.actor_optimizer = tf.keras.optimizers.Adam(LR_A)
            self.critic_optimizer = tf.keras.optimizers.Adam(LR_C)

            # Initialize target networks with same weights
            self._update_target_weights(1.0)

        def _build_actor(self):
            inputs = tf.keras.Input(shape=(self.s_dim,))
            net = tf.keras.layers.Dense(300, activation='relu')(inputs)
            outputs = tf.keras.layers.Dense(self.a_dim, activation='tanh')(net)
            scaled_outputs = tf.keras.layers.Lambda(lambda x: x * self.a_bound)(outputs)
            return tf.keras.Model(inputs, scaled_outputs)

        def _build_critic(self):
            state_input = tf.keras.Input(shape=(self.s_dim,))
            action_input = tf.keras.Input(shape=(self.a_dim,))
            concat = tf.keras.layers.Concatenate()([state_input, action_input])
            net = tf.keras.layers.Dense(300, activation='relu')(concat)
            q_value = tf.keras.layers.Dense(1)(net)
            return tf.keras.Model([state_input, action_input], q_value)

        def _update_target_weights(self, tau):
            for target_weights, eval_weights in zip(self.actor_target.weights, self.actor_eval.weights):
                target_weights.assign(tau * eval_weights + (1 - tau) * target_weights)
            for target_weights, eval_weights in zip(self.critic_target.weights, self.critic_eval.weights):
                target_weights.assign(tau * eval_weights + (1 - tau) * target_weights)

        def choose_action(self, s):
            s = s[np.newaxis, :]
            return self.actor_eval(s).numpy()[0]

        def learn(self):
            # soft target replacement
            self._update_target_weights(TAU)

            indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
            bt = self.memory[indices, :]
            bs = bt[:, :self.s_dim]
            ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
            br = bt[:, -self.s_dim - 1: -self.s_dim]
            bs_ = bt[:, -self.s_dim:]

            # 將 NumPy 陣列轉換為 TensorFlow Tensor
            bs = tf.convert_to_tensor(bs, dtype=tf.float32)
            ba = tf.convert_to_tensor(ba, dtype=tf.float32)
            br = tf.convert_to_tensor(br, dtype=tf.float32)
            bs_ = tf.convert_to_tensor(bs_, dtype=tf.float32)

            # Actor 和 Critic 更新
            with tf.GradientTape() as tape:
                a_ = self.actor_target(bs_)
                q_target = br + GAMMA * self.critic_target([bs_, a_])
                q_eval = self.critic_eval([bs, ba])
                td_error = tf.reduce_mean(tf.square(q_target - q_eval))
                critic_grads = tape.gradient(td_error, self.critic_eval.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_eval.trainable_variables))

            with tf.GradientTape() as tape:
                a = self.actor_eval(bs)
                q = self.critic_eval([bs, a])
                actor_loss = -tf.reduce_mean(q)
                actor_grads = tape.gradient(actor_loss, self.actor_eval.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_eval.trainable_variables))

        def store_transition(self, s, a, r, s_):
            transition = np.hstack((s, a, [r], s_))
            index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
            self.memory[index, :] = transition
            self.pointer += 1
            if self.pointer > MEMORY_CAPACITY:
                self.memory_full = True

        def save(self):
            self.actor_eval.save_weights('./actor_eval.weights.h5')
            self.actor_target.save_weights('./actor_target.weights.h5')
            self.critic_eval.save_weights('./critic_eval.weights.h5')
            self.critic_target.save_weights('./critic_target.weights.h5')
            print("Models saved successfully.")

        def restore(self):
            self.actor_eval.load_weights('./actor_eval.weights.h5')
            self.actor_target.load_weights('./actor_target.weights.h5')
            self.critic_eval.load_weights('./critic_eval.weights.h5')
            self.critic_target.load_weights('./critic_target.weights.h5')
            print("Models restored successfully.")
    '''
    st.code(code, language='python')
    st.divider()
    st.write("接下來為主程式main.py的介紹，在整體架構中，main.py 作為主程式統籌訓練與評估流程，初始化 rl.py 中的 DDPG 智能體，並控制每回合的學習（learn()）、數據儲存（store_transition()）、模型保存與加載（save() 和 restore()），最終通過 env.py 提供的模擬環境驅動訓練與測試。")
    st.write("""
    主程式主要控制三個參數\n
    MAX_EPISODES: 訓練過程中的最大回合數。\n
    MAX_EP_STEPS: 每回合的最大步數。 \n
    ON_TRAIN：是否訓練
    """)
    st.write("ON_TRAIN為布林值，若值為True運行訓練函式train()，若值為False運行函式eval()，了解模型訓練完的效果。")
    code = '''
    if ON_TRAIN:
        train()
    else:
        eval()
    '''
    st.code(code, language='python')



with tab5:
    st.header("2.5 streamlit UI設計與資料可視化")
    
    st.write("Streamlit 的 UI 設計是基於 Python 的簡單 API，開發者不需要學習 HTML、CSS 或 JavaScript，即可快速構建前端網頁")
    st.image("files/2_5_1.png")
    
    st.write("接下來介紹Sreamlit的基本功能")
    st.divider()
    st.write("1.添加文字到網頁")
    code = '''
    #用於顯示普通的未格式化文字
    st.text("你想要輸入的文字")
    #支援顯示文字、變量、Markdown 或其他組件
    st.write("你想要輸入的文字")
    '''
    st.code(code, language='python')
    st.divider()
    st.write("2.添加圖片到網頁")
    code = '''
    st.image("圖片的相對路徑或絕對路徑",caption="註解" width=圖片大小)
    '''
    st.code(code, language='python')
    st.divider()
    st.write("3.添加隔斷線")
    code = '''
    st.divider()
    '''
    st.code(code, language='python')
    st.divider()
    st.write("4.添加標題的方式")
    code = '''
    #添加頁面標題
    st.title("This is the Main Title")
    #添加次級標題
    st.header("This is a Header")
    #添加三級標題
    st.subheader("This is a Subheader")
    '''
    st.code(code, language='python')
    st.divider()
    st.write("5.將文字加粗、斜體、列表")
    code = '''
    #用於大標題
    st.markdown("# Main Title")
    #**用於加粗，* 用於斜體
    st.markdown("**Bold text** and *italic text*")
    #- 用於列表
    st.markdown("- Item")
    '''
    st.code(code, language='python')
    st.divider()
    st.write("6.顯示程式碼")
    code = '''
    code = 三個單引號
    
    python程式
    
    三個單引號
    st.code(code, language='python')
    '''
    st.code(code, language='python')
    st.divider()
    st.write("6.添加影片")
    code = '''
    sample_video = open("files/red_rock.mov", "rb").read()
    # Display Video using st.video() function
    st.video(sample_video, start_time = 10)
    '''
    st.code(code, language='python')
    sample_video = open("files/red_rock.mov", "rb").read()
    # Display Video using st.video() function
    st.video(sample_video, start_time = 10)
    st.divider()
    st.write("7.添加分隔頁")
    code = '''
    tab1, tab2, tab3= st.tabs(["Part 1","Part 2","Part 3"])
    with tab1:
        st.write("Part 1")
    with tab2:
        st.write("Part 2")
    with tab3:
        st.write("Part 3")
    '''
    st.code(code, language='python')

    tab1, tab2, tab3= st.tabs(["Part 1","Part 2","Part 3"])
    with tab1:
        st.write("Part 1")
    with tab2:
        st.write("Part 2")
    with tab3:
        st.write("Part 3")
    st.divider()