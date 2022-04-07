# 영화 추천:movie_camera:과 이동진 평론가의 한줄평을 함께!
[![Github All Releases](https://img.shields.io/github/downloads/junghosub/movie_recommendation/total.svg)]()
![image](https://user-images.githubusercontent.com/72376781/161991957-d3f27501-ae37-48dc-b98c-67f9fda8783d.png)


## 분석 배경 및 내용
> 일상생활을 하다보면 유튜브, 넷플릭스, 쿠팡 등 많은 부분에서 추천 알고리즘의 도움을 받게 됩니다. 저도 평소 영화를 고를 때 추천 알고리즘의 도움을 많이 받지만 이동진 평론가의 도움도 자주 받습니다. 그러던 중 <b>이동진 평론가가 남긴 평점과 한줄평을 함께 보는 영화 추천 시스템을 만들어 보고 싶어 해당 프로젝트를 진행하게 되었습니다.</b>

1. ``네이버 블로그, 네이버 API와 Kaggle을 통해 데이터 수집``을 진행했습니다. 데이터는 영화 제목(한/영), 개봉년도, 장르, 네이버 평점, IMDB 평점, 해당 평점에 투표한 IMDB 유저의 수, 이동진 평론가의 평점, 감독, 주연 배우, 이동진 평론가의 한줄평으로 구성되어있습니다.
2. ``특이값 분해와 코사인 유사도를 활용한 하이브리드 기반의 추천 알고리즘``을 채택하였습니다. 초기엔 코사인 유사도를 활용한 콘텐츠 기반의 추천 필터링만을 적용했습니다. 하지만 개인화된 추천과 다양한 추천 목록을 제공하고 싶어 하이브리드 기반의 추천 알고리즘으로 변경하게 되었습니다.
3. ``유저의 ID``와 ``영화 제목``이 주어지면 ``해당 유저의 취향에 알맞는 영화를 10개 추천``해줍니다.

## 데이터 수집

- 이동진 평론가의 평점과 한줄평: [네이버 블로그](https://soulmatt.tistory.com/entry/%EC%9D%B4%EB%8F%99%EC%A7%84-%ED%8F%89%EB%A1%A0%EA%B0%80-%ED%95%9C%EC%A4%84%ED%8F%89-%EB%AA%A8%EC%9D%8C-%EC%B4%9D-1115%ED%8E%B8)를 통해 데이터(2019년까지) 수집
- 영화 감독, 주연 배우, 네이버 영화 평점: 네이버 API를 통해 수집
- 영화 장르, IMDB 영화 평점, 해당 영화의 추천인 수: Kaggle의 The Movie Dataset을 통해 수집(2017년까지)
- 협업 필터링을 위한 유저 정보: Kaggle의 The Movie Dataset을 통해 700명의 유저 정보 수집
- 해당 데이터들을 join하여 약 5,500개 가량의 데이터를 수집함

![image](https://user-images.githubusercontent.com/72376781/162008581-08ee22e9-d180-4996-a5e3-f8614b15c5d5.png)

## 주요 코드

<details>
  <summary> 1. 코사인 유사도를 구하기 위한 <b>전처리</b> </summary>
  
```python
def preprocess(df):
    # 장르, 감독, 배우의 결측치는 공백으로 처리
    df['genres'].fillna('', inplace = True)
    df['director'].fillna('', inplace = True)
    df['actor'].fillna('', inplace = True)
    
    # 감독은 2배의 가중치를 줌.
    df['director'] = df['director'].apply(lambda x: [x, x])
    df['director'] = df['director'].apply(lambda x : (' ').join(x))
    
    # 주연 배우도 2배의 가중치.
    df['actor'] = df['actor'].apply(lambda x: [x, x])
    df['actor'] = df['actor'].apply(lambda x : (' ').join(x))

    # 장르, 감독, 배우에 기반한 코사인 유사도를 구하기 위해 content 컬럼 생성
    df['content'] = df['genres'] + ' ' + df['director'] + ' ' + df['actor']
    
    return df
```
</details>

<details>
  <summary> 2. <b> 코사인 유사도 구하기 </b> </summary>

```python
# CountVectorizer를 통해 단어의 빈도 수 구하기
count_vect = CountVectorizer(min_df=0, ngram_range=(1,2))
content_mat = count_vect.fit_transform(md['content'])

# 코사인 유사도 구한 후 정렬
cosine_sim = cosine_similarity(content_mat, content_mat)
content_sim_sorted_ind = cosine_sim.argsort()[:, ::-1]
```

</details>

<details>
  <summary> 3. <b>가중 평점</b> 구하기 </summary>

평점은 오류가 발생할 여지가 있습니다. 1명만 투표한 10점짜리 영화와 5,000명이 투표한 8점짜리 영화 중 어떤 영화를 신뢰할 수 있을까요? 5,000명이 투표한 8점짜리 영화가 더욱 신뢰성이 있을 것입니다. 이후 유저에게 추천할 때, 평점순 정렬을 통해 추천하게 됩니다. 평점은 매우 높지만 <b>투표 참여 수가 매우 낮은 영화들이 추천에 등장하는 것을 방지하기 위해 투표 수를 고려한 가중 평점을 구합니다.</b>
  
![image](https://user-images.githubusercontent.com/72376781/162013845-60263327-3f0e-4eab-b99b-4c92e6d02f74.png)

하지만 현재 데이터에는 IMDB 평점과 투표 참여 수가 누락된 데이터들이 존재합니다. 결측치가 없는 네이버 평점을 사용할까 했지만 편차가 컸고 유명하지 않음에도 평점이 매우 높은 영화들이 존재했었습니다.

<figure class="half">
    <img src= "https://user-images.githubusercontent.com/72376781/162015266-5f1214a2-faf2-4f0b-802c-af37198efff1.png" width = "350">
    <img src= "https://user-images.githubusercontent.com/72376781/162021603-b659e87e-9816-466c-bdb0-11c91679852f.png" width = "350">
</figure>

세 평점의 분포가 조금씩 차이나는 것을 확인할 수 있습니다.
  
- 네이버: 오른쪽으로 치우쳐진 분포를 볼 수 있습니다. 실제로 네이버의 평점은 다른 서비스들보다 높은 점수대를 이루고 있습니다. 또한 네이버 API를 통해 데이터를 수집하는 과정에서 옛날 영화들 같은 경우엔 평점이 제대로 표기되지 않는 것도 많았습니다.
- IMDB: 네이버와 비교하면 상대적으로 정규분포 모형을 띄고 있습니다. 하지만 평균적인 점수대에 많은 데이터가 몰려 있는 것을 확인할 수 있습니다.
- 이동진 평론가의 평점: 이동진 평론가는 5점 스케일로 점수를 부여했습니다. 이를 다른 평점과 마찬가지로 10점 스케일로 보기 위해 2배곱을 해주었습니다. 어느정도 정규분포를 이루고 있지만 결측치가 많아서 기준 평점으로 사용하긴 힘들 것 같습니다. 

결과적으로 IMDB 평점이 누락된 경우 다른 방식을 적용하기로 하였습니다. 네이버 평점 - IMDB 평점 = 1.3입니다. 또한 위의 가중 평점 공식을 적용하여 기존 IMDB 평점과 가중 평점의 차이 평균를 확인해보니 0.077이었습니다. 다시 말해 가중 평점을 적용하면 평균적으로 기존 평점에서 -0.077이 떨어진 평점을 가진다는 것입니다. 이를 고려하여 네이버 평점에서 1.4를 빼주는 처리를 해줄 수 있습니다. 하지만 최대값에서 많은 차이가 났기 때문에 보수적으로 -1.45를 빼주었습니다.
  
``` python
# 가중 평점을 구하는 함수
def weighted_vote_average(record):

    # IMDB 평점이 누락된 경우
    if np.isnan(record['imdb rating']):
        return round(record['naver rating'] - 1.45, 2)
    
    # IMDB 평점이 누락되지 않은 경우
    else: 
        percentile = 0.6
        m = md['vote_count'].quantile(percentile)
        C = md['imdb rating'].mean()
        v = record['vote_count']
        R = record['imdb rating']
    
        return round(((v/(v+m)) * R ) + ( (m/(m+v)) * C ), 2)
  
# 위의 함수를 적용하여 가중 평점 구하기
md['weighted_rating'] = md.apply(weighted_vote_average, axis=1)
```

</details>

<details>
  <summary> 4. <b>콘텐츠 기반 영화 추천</b> </summary>

 이제 코사인 유사도를 통해 콘텐츠 기반 영화 추천을 해보겠습니다.
  
```python
# 코사인 유사도가 높은 영화 10선 추천
def recom_movie(df, sorted_ind, title_name, top_n=10):
    #  해당 영화 제목의 index 추출
    title_movie = df[df['title'] == title_name]
    title_index = title_movie.index.values
    
    # top_n의 2배에 해당하는 유사성이 높은 index 추출 
    similar_indexes = sorted_ind[title_index, :(top_n*2)]
    similar_indexes = similar_indexes.reshape(-1)
    
    # 기준 영화 index는 제외
    similar_indexes = similar_indexes[similar_indexes != title_index]
     
    # top_n의 2배에 해당하는 후보군에서  rating이 높은 순으로 top_n 만큼 추출 
    similar_movies =  df.iloc[similar_indexes].sort_values('weighted_rating', ascending=False)[:top_n][['title', 'publication date', 'imdb rating', 'vote_count', 'weighted_rating','critic_rating', 'review']]
    similar_movies.rename(columns = {'title' : '제목', 'publication date' : '개봉년도', 'weighted_rating' : '가중 평점', 'critic_rating' : '이동진 평점', 'review' : '이동진의 한줄평'}, inplace = True)
    
    return similar_movies[['제목', '개봉년도', '가중 평점', '이동진 평점', '이동진의 한줄평']]

# 기생충과 유사한 영화 10개
similar_movies = recom_movie(md, content_sim_sorted_ind, '기생충',10)
similar_movies
```
<img src = "https://user-images.githubusercontent.com/72376781/162022980-7336f421-d0f9-43b7-9259-f6958d762ac0.png" width ="700">

'기생충'의 감독인 봉준호, 주연 배우였던 송강호가 출연했던 영화들을 위주로 추천하는 것을 확인할 수 있었습니다. 꽤나 괜찮은 영화 목록들을 추천해주네요! 가중 평점 기준으로 6.55가 넘으면 상위 20%에 해당되는 수작으로 평가되며 6.96이 넘으면 상위 10%인 명작으로 분류됩니다.

</details>

<details>
  <summary> 5. <b>하이브리드 기반 추천</b> </summary>

콘텐츠 기반 필터링을 통해서 꽤나 괜찮은 영화 목록들을 추천해주는 것을 확인할 수 있었습니다. 하지만 콘텐츠 기반 필터링에도 단점이 존재합니다. 아무래도 메타 데이터를 통해 추천하기 때문에 개인화되고 다양한 형식의 항목을 추천하기 어려운 단점이 있습니다. 이를 위해 협업 필터링을 함께 적용해 하이브리드 기반의 개인 맞춤 추천을 구현해볼 것입니다.

```python
# 특이점 분해 방식을 활용
svd = SVD()
cross_validate(svd, data, measures = ['RMSE', 'MAE'], cv = 5, verbose = True)
```

Evaluating RMSE, MAE of algorithm SVD on 5 split(s).
|평가 지표 및 시간|Fold 1|Fold 2|Fold 3|Fold 4|Fold 5|Mean|Std|
|---|---|---|---|---|---|---|---|
|RMSE (testset)    |0.8925|  0.8963|  0.9026|  0.8994|  0.8978|  0.8977|  0.0033|  
|MAE (testset)     |0.6881|  0.6890|  0.6928|  0.6914|  0.6932|  0.6909|  0.0020|  
|Fit time          |3.77|    3.80|    3.78|    3.77|    3.78|    3.78|    0.01|    
|Test time         |0.09|    0.19|    0.11|    0.20|    0.11|    0.14|    0.05|    

```python
# 해당 유저가 본 영화의 장르들을 시각화 해주는 함수
def make_plot(user_id):
    try:
        plt.figure(figsize = (8,6))
        # 해당 유저 ID 데이터만 보기
        user = ratings.loc[ratings['userId'] == user_id]
        user = user.merge(md, left_on = 'movieId', right_on = 'id')
        
        # 각 영화별 장르를 카운팅 하기 위해 split()
        user = user['genres'].apply(lambda x : x.split())
        
        # 이중 for문을 통해 각 리스트에 해당하는 영화 장르 추출
        array = []
        for items in user:
            for item in items:
                array.append(item)

        user_view = pd.DataFrame()
        user_view['genres'] = array

        # 시각화
        count_df = pd.DataFrame(user_view.groupby('genres')['genres'].count().sort_values(ascending = False))
        sns.barplot(data = count_df, x = count_df.index, y = count_df['genres'], palette = 'Paired')
        plt.xticks(rotation = 90)
        plt.xlabel('장르')
        plt.ylabel('영화를 본 횟수')
        plt.title('User ID: ' + str(user_id), size = 14)
        plt.show()
    except:
        print('없는 유저입니다. 다른 유저 ID를 검색해주실래요?😥')
```
  
<figure class="third">
    <img src= "https://user-images.githubusercontent.com/72376781/162029392-5eba445e-4047-40b2-9f5f-986912f05408.png" width = "400">
    <img src= "https://user-images.githubusercontent.com/72376781/162026834-f0fea2ae-a10a-49fc-a230-2d22a6bbc9d8.png" width = "400">
</figure>

- 236번 유저: 스릴러 영화를 가장 많이 시청했고 드라마와 액션, 범죄 등의 장르를 좋아하는 것 같습니다.
- 73번 유저: 드라마 영화를 특히 많이 본 유저입니다. 
  
```python
# 하이브리드 기반의 영화 추천
def hybrid(userId, title):
     try:
        # 해당 영화 제목에 맞는 movieId 추출
        idx = indices[title]
        tmdbId = id_map.loc[title]['id']
        movie_id = id_map.loc[title]['movieId']
        
        # 해당 영화와 높은 코사인 유사도 점수
        sim_scores = list(enumerate(cosine_sim[int(idx)]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:20]
        movie_indices = [i[0] for i in sim_scores]

        # 예상 평점을 기준으로 상위 10개 영화 추천
        movies = md.iloc[movie_indices][['id', 'title', 'publication date', 'weighted_rating', 'naver rating', 'critic_rating', 'vote_count', 'review']]
        movies['est'] = round(movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est), 1)
        movies['exp_rating'] = round((movies['est'] * 2 + movies['weighted_rating']) / 2 , 2)
        movies = movies.sort_values(by = 'exp_rating', ascending=False)

       # 후보군 중  exp_rating이 높은 순으로 10개의 영화 추천 
        movies.rename(columns = {'title' : '제목', 'publication date' : '개봉년도', 'exp_rating' : '예상 평점', 'critic_rating' : '이동진 평점', 'review' : '이동진의한줄평'}, inplace = True)
        return movies[['제목', '개봉년도', '예상 평점', '이동진 평점', '이동진의 한줄평']].head(10)
  
     except:
            print('유저 ID 또는 영화 제목을 다시 한 번 확인해주실래요? 🥺')
```

236번 유저의 추천 목록
```python
hybrid(236, '기생충')
```
<img src = "https://user-images.githubusercontent.com/72376781/162141022-5065e320-c761-4fa2-af52-d5a7e7ac4c41.png" width = "550">

73번 유저의 추천 목록
```python
hybrid(73, '기생충')
```
<img src = "https://user-images.githubusercontent.com/72376781/162141072-f1008a54-b345-4940-a3c9-633fba0b0f06.png" width = "550">

장르만을 확인해보았지만 두 유저의 데이터를 기반으로 기생충과 비슷한 영화를 추천해달라고 했습니다. 각 영화에 따라 예상 평점이 다르게 나온 것을 보아 해당 유저의 성향을 반영한 추천이 이루어진 것으로 보입니다.  
</details>

## 느낀 점 및 보완점
- 이번 프로젝트를 통해 ``직접 프로젝트를 위한 데이터를 수집 및 구축``하고 ``추천 시스템의 원리에 대해 이해``할 수 있었습니다.
-  ``데이터의 중요성에 대해 다시 한 번 느꼈습니다.`` 콘텐츠 기반 필터링은 메타 데이터를 통해 추천하게 됩니다. 이를 위해 더욱 세분화된 메타 데이터가 있으면 좋겠다고 생각했습니다. 실제로 넷플릭스에서는 태거 Tagger라는 일자리가 있습니다. 해당 콘텐츠의 메타 데이터를 아~주 세분화하여 태깅한다고 합니다. 본 분석에서 진행했던 장르, 감독, 주연 배우 뿐만이 아닌 ``줄거리, 분위기, 등장 인물의 특성 등 세분화된 메타 데이터베이스를 구축한다면 더욱 좋은 분석이 진행``될 것 같습니다.
- 유저의 시청 기록, 유저가 각 영화에 매긴 평점 데이터가 필요없었지만 협업 필터링을 함께 적용한 하이브리드 기반의 경우에는 유저 데이터가 필수적이었습니다. 이를 위해 Kaggle의 데이터를 활용했지만 4년 전 데이터라 아쉬움이 많이 남습니다. ``실제 현업에서 일하면서 유저들의 데이터들을 분석한다면 더욱 가치 있는 분석 결과가 나올 것 같습니다.``
