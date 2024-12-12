import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
import numpy as np
import plotly.graph_objects as go

# 감정 분석을 위한 감정 카테고리 및 키워드
EMOTION_KEYWORDS = {
    '기쁨': ['좋아', '행복', '기뻐', '신나', '즐거워', '설레', '웃', '감사', '만족'],
    '슬픔': ['슬퍼', '우울', '눈물', '속상', '괴로워', '외로워', '힘들어', '지쳤어', '아파'],
    '분노': ['화나', '짜증', '열받아', '미워', '싫어', '실망', '후회', '억울', '불만'],
    '불안': ['걱정', '불안', '두려워', '무서워', '긴장', '초조', '혼란', '고민', '망설여']
}

EMOTION_COLORS = {
    '기쁨': '#FFD700',     # 밝은 노랑
    '슬픔': '#4682B4',     # 파랑
    '분노': '#FF4500',     # 붉은 주황
    '불안': '#9370DB'      # 보라
}

EMPATHY_MESSAGES = {
    '기쁨': ['기쁜 마음이 느껴지네요. ', '행복해 보이셔서 저도 기뻐요. ', '그 기쁨을 함께 나눌 수 있어 좋네요. '],
    '슬픔': ['많이 힘드시군요. ', '그런 감정을 느끼시는 게 당연해요. ', '함께 이야기를 나누고 싶어요. '],
    '분노': ['화가 나시는 게 당연해요. ', '그런 상황에서 화가 나시겠어요. ', '속상한 마음이 이해됩니다. '],
    '불안': ['걱정이 많으시군요. ', '불안한 마음이 느껴져요. ', '그런 걱정을 하시는 게 자연스러워요. ']
}

@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('wellness_dataset.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

def analyze_emotion(text):
    """키워드 기반 감정 분석"""
    text = text.lower()
    emotions_found = []
    
    for emotion, keywords in EMOTION_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            emotions_found.append(emotion)
    
    if emotions_found:
        return np.random.choice(emotions_found)  # 여러 감정이 감지되면 그 중 하나를 선택
    return '중립'  # 감지된 감정이 없을 경우

def get_empathy_response(emotion):
    """감정별 맞춤 공감 메시지 생성"""
    if emotion in EMPATHY_MESSAGES:
        return np.random.choice(EMPATHY_MESSAGES[emotion])
    return '그런 감정을 느끼시는군요. '

def create_donut_chart(emotion_counts):
    """감정 도넛 차트 생성"""
    colors = [EMOTION_COLORS.get(emotion, '#808080') for emotion in emotion_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=list(emotion_counts.index),
        values=list(emotion_counts.values),
        hole=.3,
        marker_colors=colors
    )])
    
    fig.update_layout(
        title_text="감정 분석 결과",
        showlegend=True,
        width=400,
        height=400
    )
    
    return fig

def get_conversation_history():
    """대화 히스토리를 문자열로 반환"""
    history = []
    for i in range(len(st.session_state['past'])):
        history.append(f"사용자: {st.session_state['past'][i]}")
        history.append(f"감정: {st.session_state.emotion_history[i]}")
        if len(st.session_state['generated']) > i:
            history.append(f"챗봇: {st.session_state['generated'][i]}")
        history.append("-" * 50)
    return "\n".join(history)

def save_conversation():
    """대화 내용을 파일로 저장"""
    history = get_conversation_history()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"chat_history_{timestamp}.txt", "w", encoding="utf-8") as f:
        f.write(history)

# 메인 애플리케이션
model = cached_model()
df = get_dataset()

st.header('맞춤형 심리상담 챗봇')
st.markdown("[딥러닝 기말 발표_김민정]")

# 세션 상태 초기화
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'emotion_history' not in st.session_state:
    st.session_state['emotion_history'] = []

# 사이드바에 설정 추가
st.sidebar.header("설정")
response_length = st.sidebar.slider("응답 길이", 1, 5, 1)
empathy_level = st.sidebar.select_slider(
    "공감 수준",
    options=["낮음", "중간", "높음"]
)

# 메인 채팅 인터페이스
with st.form('form', clear_on_submit=True):
    user_input = st.text_input('당신: ', '')
    cols = st.columns([1, 1])
    submitted = cols[0].form_submit_button('전송')
    save_button = cols[1].form_submit_button('대화 저장')

if submitted and user_input:
    # 감정 분석
    current_emotion = analyze_emotion(user_input)
    st.session_state.emotion_history.append(current_emotion)
    
    # 임베딩 및 응답 생성
    embedding = model.encode(user_input)
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    
    # 상위 응답 선택
    top_responses = df.nlargest(response_length, 'distance')
    answer = ' '.join(top_responses['챗봇'].tolist())
    
    # 감정 분석 결과를 기반으로 응답 수정
    emotion_counts = pd.Series(st.session_state.emotion_history).value_counts()
    dominant_emotion = emotion_counts.index[0]  # 가장 빈도가 높은 감정
    
    # 공감적 응답 생성
    empathy_response = get_empathy_response(current_emotion)
    if empathy_level == "높음":
        answer = empathy_response + answer
    elif empathy_level == "중간":
        if np.random.random() > 0.5:  # 50% 확률로 감정 언급
            answer = empathy_response + answer
    
    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer)

if save_button:
    save_conversation()

# 감정 분석 도넛 차트 표시
if st.session_state.emotion_history:
    emotion_df = pd.DataFrame(st.session_state.emotion_history, columns=['감정'])
    emotion_counts = emotion_df['감정'].value_counts()
    
    # 도넛 차트 생성 및 표시
    fig = create_donut_chart(emotion_counts)
    st.plotly_chart(fig)

# 대화 표시
for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_bot')
