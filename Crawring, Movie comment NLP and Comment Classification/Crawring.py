#pip install konlpy
#pip install bs4
#pip install requests

import requests
from bs4 import BeautifulSoup
from konlpy.tag import Okt
import pandas as pd
import numpy as np
import re

################    크롤링  #################

twitter = Okt()
url = "https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?" \
      "code=181414&type=after&isActualPointWriteExecute=false&" \
      "isMileageSubscriptionAlready=false&isMileageSubscriptionReject=false&page={}"


# 클리닝 함수
def clean_text(text):
    cleaned_text = re.sub('[a-zA-Z]', '', text)
    cleaned_text = re.sub('[\{\}\[\]\/?.,;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"]',
                          '', cleaned_text)
    return cleaned_text

def get_reple(page=1):
    response = requests.get(url.format(page))
    soup = BeautifulSoup(response.text, 'html.parser')
    s, t = [], []

    for li in soup.find('div', {'class': 'score_result'}).find_all('li'):
        # text를 가져올때 공백을 제거한다
        str = li.p.get_text(" ", strip=True)
        str = clean_text(str)

        # 8점이상 좋은영화
        if int(li.em.text) >= 8:
            s.append(1)
            t.append(str)
        # 5점 이하 좋지 않은영화
        elif int(li.em.text) <= 5:
            s.append(0)
            t.append(str)
    return s, t


score, text = [], []

# 1~150page까지 불러온 후 배열에 넣어준다.
for i in range(1, 150):
    print('요청 횟수:', i)
    s, t = get_reple(i)
    # 배열을 합하여준다.
    score += s
    text += t

df = pd.DataFrame([score, text]).T
df.columns = ['score', 'text']
# csv파일로 저장한다.
df.to_csv('test.csv', encoding='utf-8-sig')

################    여기까지 크롤링  #################

###############     LogisticRegression을 이용한 감정분석 with TF-IDF,BOW(TfidfVectorizer사용)    #################
# TF-IDF란? https://nesoy.github.io/articles/2017-11/tf-idf
# BOW란? url1 :: https://medium.com/@omicro03/%EC%9E%90%EC%97%B0%EC%96%B4%EC%B2%98%EB%A6%AC-nlp-7%EC%9D%BC%EC%B0%A8-bag-of-words-20b2af01d56f
#        url2 :: https://ldabook.com/word-representations.html
# TfidfVectorizer란? https://datascienceschool.net/view-notebook/3e7aadbf88ed4f0d87a76f9ddc925d69/
# df = pd.read_csv('test.csv', encoding='utf-8')
# x값: bow로 벡터화한값, y: 영화 평점
text = df.get('text')
score = df.get('score')

# 트레이닝/테스트셋을 분리한다.
# random_state: random으로 나누어준다, 0은 항상 같은결과가 나오도록 seed고정
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(text, score, test_size=0.2, random_state=0)
train_y = train_y.astype('float')
test_y = test_y.astype('float')
type(train_x)
print(train_x)
print(train_y)

# 단어들을 모아 빈도수 체크 (TF-IDF)
# feature(문장의 특징) 노출수를 가중치로 설정한 Bag Of Word 벡터를 만든다.
from sklearn.feature_extraction.text import TfidfVectorizer

# ngram으로 단어를 n개로 잘라서 보겠다는 의미 (1, 2그램 사용했다)
# ex) 재밌다. 너무 재밌다. 노잼. 완전 노잼.
# tokenizer는 twitter.morphs사용
# max_df: 너무 많은단어들(the, a)제외
# min_df: 너무적은단어 제외 몇만개에서 3개만 나오는 단어의 경우 의미가 없을 가능성이 높다
# https://datascienceschool.net/view-notebook/3e7aadbf88ed4f0d87a76f9ddc925d69/
tfv = TfidfVectorizer(tokenizer=twitter.morphs, ngram_range=(1, 2), min_df=3, max_df=0.9)
tfv.fit(train_x)
tfv_train_x = tfv.transform(train_x)
tfv_train_x

## 참고(없어도 학습가능) tf-idf 형태 ##
#  tfidf 값을 가중치로 설정한 행렬을 볼 수 있다.
# 부정예시 : 돈 아깝다
features = tfv.get_feature_names()
stm = np.asarray(tfv_train_x.toarray())
df = pd.DataFrame(stm)
df.columns = features
print(df)

# 이진분류 알고리즘(로지스틱 회귀)
from sklearn.linear_model import LogisticRegression
# 하이퍼파라미터 최적화
from sklearn.model_selection import GridSearchCV

clf = LogisticRegression(random_state=0)
params = {'C': [0.1, 1, 0.01, 3, 5, 10]}

# params중 정확도가 가장 높은 param을 선택
# cv=4 : 4번의 모의를 통해 가장 좋은 params를 선택
# https://datascienceschool.net/view-notebook/ff4b5d491cc34f94aea04baca86fbef8/ 참조
# verbose: 상세정보(숫자가 클수록 상세정보가 자세히 보여진다)
grid_cv = GridSearchCV(clf, param_grid=params, cv=4, scoring='accuracy', verbose=1)

# 최적의 하이퍼파라미터를 사용하여 학습
grid_cv.fit(tfv_train_x, train_y)
# grid_cv: 학습한 모델에 접근할 수 있도록 PREDICT와 SCORE 메서드 제공

# 가장 좋은 parameter
print(grid_cv.best_params_)
# 가장 좋은 parameter
print(grid_cv.best_score_)


# test data 학습

# 벡터변환기
tfv_test_x = tfv.transform(test_x)
# 정확도 예측 (train data와 test data의 정확도를 비교하며 over fitting에 빠졌는지 확인해준다)
grid_cv.best_estimator_.score(tfv_test_x, test_y)

# 만들어진 모델로 영화 긍부정 구분

# 긍정 테스트
#ex = ['진짜 재밌어요! 강추!!']

# 부정 테스트
ex = ['개 노잼 ㅡㅡ']
# 감정분석사전과는 달리 비속어도 구분이 가능하다
# 하지만 라벨링이 되어있지않은 소셜데이터의경우 감정분석사전을 사용하는게 더 좋을 수 있다.
my_review = tfv.transform(ex)

# 0: 부정, 1: 긍정
print(grid_cv.best_estimator_.predict(my_review))

## 참고 (없어도 학습가능) 입력 tf-idf 형태 ##
features = tfv.get_feature_names()
stm = np.asarray(my_review.toarray())
df = pd.DataFrame(stm)
df.columns = features
print(df)