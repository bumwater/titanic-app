import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

fm.fontManager.addfont("NanumGothic.ttf")
font_name = fm.FontProperties(fname="NanumGothic.ttf").get_name()
print(font_name)
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False

st.title("타이타닉 생존 분석")
st.markdown("> 타이타닉 사건의 생존율에 성별이 가장 큰 영향을 미쳤고 그다음이 객실 등급과 나이였다.")



def get_title(name):
    step1 = name.split(",")
    step2 = step1[1]
    step3 = step2.split(".")
    step4 = step3[0]
    step5 = step4.strip()
    return step5

@st.cache_data
def load_data():
    print("CSV 읽는 중...")
    df = pd.read_csv("titanic.csv")
    df["Title"] = df["Name"].apply(get_title)
    title_median = df.groupby("Title")["Age"].median()
    df["Age"] = df["Age"].fillna(df["Title"].map(title_median))
    df = df.drop(columns=["Cabin"])
    df = df.dropna(subset=["Embarked"])

    return df

st.header("데이터")
st.write("타이타닉 승객 891명의 나이, 성별,객실등급,생존여부 데이터")

df = load_data()
st.dataframe(df.head(10))

st.header("전처리")
st.markdown("""
결측치가 있는 컬럼: 나이(177명), 탑승 항구(2명), 객실 번호(687명)

- 객실 번호: 77%가 비어 있어 열을 삭제했다.
- 탑승 항구: 2명뿐이라 해당 행을 삭제했다.
- 나이: 처음에는 전체 중앙값(28세)으로 채워 그래프를 그려보니 28세에 177명이 몰려 분포가 왜곡됐다.
  그래서 이름에서 호칭(Mr, Miss, Master 등)을 추출해 호칭별 중앙값으로 다시 채웠다.
  그 결과 남자아이(Master)는 3.5세로 채워졌다. 다만 Mr 119명이 30세로 채워져 봉우리가 완전히 없어지진 않았다.
""")

st.header("분석")

pclass = st.selectbox("객실 등급을 고르세요", [1,2,3])
sex = st.selectbox("성별 고르세요", ["male" , "female"])

filtered = df[(df["Pclass"] == pclass) & (df["Sex"] == sex)]
rate = filtered["Survived"].mean()
st.write(f"{pclass}등석 {sex}의 생존율 {rate:.1%}")

fig, ax = plt.subplots()
sns.barplot(data=df, x="Sex", y="Survived", ax=ax)
ax.set_title("성별 생존율")
st.pyplot(fig)
st.write("여성이 남성에 비해 훨씬 많이 생존하였다.")

fig, ax = plt.subplots()
sns.barplot(data=df, x="Pclass", y="Survived", hue="Sex" ,ax=ax)
ax.set_title("객실등급과 성별에 따른 생존율")
st.pyplot(fig)
st.write("객실 등급이 높을 수 록 많이 생존하였다. 하지만 3등급 여성과 1등급 남성의 생존율을 비교했을때 3등급 여성이 더 많이 생존하였다.")

fig, ax = plt.subplots()
sns.histplot(data=df, x="Age", hue="Survived", bins=30, multiple="stack", ax=ax)
ax.set_title("나이별 생존 여부")
st.pyplot(fig)
st.write("10세 이하 나이대 생존률이 높았다.")

st.header("결론")
st.markdown("""
- 10세 이하 어린이의 생존율이 다른 나이대보다 눈에 띄게 높았다. 그 외 나이대는 일정한 경향이 없었다..
- 여성의 생존율이 남성보다 훨씬 높았다.
- 객실 등급이 높을수록 생존율이 높았다. 1등석 -> 3등석으로 갈수록 낮아졌다.
- 객실 등급 보다 성별의 영향이 더 컸다. 3등석 여성이 1등석 남성보다 생존율이 높았다.
""")