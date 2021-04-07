# Keras-CNN-Dog-or-Cat-Classification
2nd problem of assignment "차량지능기초"

MNIST 분류기 코드와 아주 유사하게 진행된다.

![image](https://user-images.githubusercontent.com/81463668/113806783-ed30f380-979d-11eb-913d-c4f818a37da4.png)

②의 MNIST코드에서 말했듯 여러 library들과 keras framework를 가져와준다.

![image](https://user-images.githubusercontent.com/81463668/113806794-f1f5a780-979d-11eb-898b-8b513200364a.png)

이번에도 마찬가지로 빠르게 학습할 때 쓰는 FASTRUN은 사용하지 않고 이미지에 대한 스펙을 입력해준다. channel은 input 이미지가 RGB image이기 때문에 3으로 넣어주었다.

![image](https://user-images.githubusercontent.com/81463668/113806808-f91cb580-979d-11eb-98cf-c65904c7fa58.png)

train dataset을 정리하기 위해 가져와서 for문으로 .으로 모두 나눈다음 category 이름이 ‘dog’이거나 그렇지않은(‘cat’) 경우로 나누어주어 label을 분리시킨다.

![image](https://user-images.githubusercontent.com/81463668/113806816-fde16980-979d-11eb-86d1-c6cc77f7c735.png)
![image](https://user-images.githubusercontent.com/81463668/113806821-ffab2d00-979d-11eb-83ec-a7110f39443b.png)

df.head, df.tail 함수를 이용해 dataset의 앞부분과 뒷부분의 전처리가 어떻게 되었는지 확인해준다.

![image](https://user-images.githubusercontent.com/81463668/113806839-076ad180-979e-11eb-846f-c9395c6fef49.png)

이미지의 라벨들의 개수를 확인할 수 있다. (1은 dog, 0은 cat)

![image](https://user-images.githubusercontent.com/81463668/113806846-0b96ef00-979e-11eb-9f04-8e319c6df078.png)

matplot을 이용해 sample image를 가져와서 보여준다.

![image](https://user-images.githubusercontent.com/81463668/113806851-0f2a7600-979e-11eb-83a4-af39f001890a.png)

강아지의 사진이 plot으로 잘 구현된 것을 확인할 수 있다.

![image](https://user-images.githubusercontent.com/81463668/113806860-13569380-979e-11eb-8fe3-fa16917c836f.png)

MNIST에서 했던 것과 같이 keras의 Sequential함수를 사용해서 신경망 층을 넣어주고 한번에 처리할 것이 예상된다. model은 Sequential의 객체이다.
첫 번째 conv층에서는 3x3 filter 32개를 사용하고 relu 활성화함수를 쓴 후 batchnormalization기법을 한번 사용해준다. 이 후 maxpooling층과 dropout(0.25) 기법으로 overfitting을 방지하는 부분도 보인다. 다음 conv층은 첫 번째 층과 동일하되 filter의 개수가 64개 세 번째 conv층은 동일하되 filter의 개수가 128개이다.
마지막으로 Flatten함수를 통해 전까지 계산한 tensor를 1차원화시켜서 Dense함수를 통해 512개의 노드와 FC층을 이루게 한다. 이후 Batchnormalization기법과 dropout을 앞에서와 같이 사용하고 마지막으로 2개의 노드와 Dense함수를 써서 신경망을 거쳐 softmax함수까지 거치면 binaryclassification에 걸맞는 0또는 1값이 나타난다.
마지막코드인 model.summary는 각 층에서의 dimension과 parameter 개수를 나타내준다.
![image](https://user-images.githubusercontent.com/81463668/113806877-1a7da180-979e-11eb-836b-a4339bf883b2.png)
![image](https://user-images.githubusercontent.com/81463668/113806892-1ea9bf00-979e-11eb-93f6-24ec6741c95e.png)
![image](https://user-images.githubusercontent.com/81463668/113806897-1fdaec00-979e-11eb-9d96-8cac76645a75.png)
![image](https://user-images.githubusercontent.com/81463668/113806900-21a4af80-979e-11eb-9d90-3dd6fce707e6.png)

만약 training 도중에 런타임이 끊긴다거나 오류가 발생할 수 있기 때문에 학습중간까지 학습된 정보들을 저장하고 가져와줄 수 있는 tool을 마련한다.
early stopping의 본래기능은 너무 train set에 많이 학습되면 cost가 낮아지다가 overfitting 되면서 cost가 되려 높아지는데 이렇게 높아지기 전에 학습을 끝내버린다.
learning_rate_reduction은 학습진행중에 lr이 작아지도록 설정해준 부분이고 학습중에 상태창에 나타나도록 되어있다.

![image](https://user-images.githubusercontent.com/81463668/113806916-26696380-979e-11eb-80a5-8e3b26024019.png)

category에서 1과 0을 가지고있었던 부분을 cat과 dog라는 string으로 바꿔준다.
사이킷런을 통해 train과 valid set으로 분할하고 기존인덱스를 버리고 재배열한다.

![image](https://user-images.githubusercontent.com/81463668/113806931-2c5f4480-979e-11eb-8b53-ac8d6cca26ed.png)
train과 valid의 총개수를 빼내주고 mini batch size를 정해준다.

![image](https://user-images.githubusercontent.com/81463668/113806946-32edbc00-979e-11eb-991e-33319cd83c83.png)

![image](https://user-images.githubusercontent.com/81463668/113806948-341ee900-979e-11eb-80f7-59578f180ccf.png)

train과 valid 의 data를 만들어주고 data에 대한 설정도 약간 변경한다. ( data 증폭 )

![image](https://user-images.githubusercontent.com/81463668/113806958-38e39d00-979e-11eb-929b-e14533f2377a.png)

matplot으로 data증폭한 image들을 갖고와서 보면 한 장의 image 다양한 image들을 만들어 냈다. (data augmentation 기법)

![image](https://user-images.githubusercontent.com/81463668/113806968-400aab00-979e-11eb-90f3-3efab4541710.png)

FAST_RUN일 경우에는 epoch을 3으로 정하고 아니면 50으로 정한다.
model을 기존에 정한 hyperparameter들로 설정하고 학습시키면서 history안에 넣어준다.

![image](https://user-images.githubusercontent.com/81463668/113806973-43059b80-979e-11eb-9010-346ebe8f37b8.png)

학습이 진행되면서 loss의 값은 줄어들고 정확도는 올라가는 것을 볼 수 있다.
loss에 대한 부분이 아주 많이 줄어들지는 않지만 다른 다양한 기법들을 사용하면 줄일 수 있을 것으로 본다.

![image](https://user-images.githubusercontent.com/81463668/113806985-4862e600-979e-11eb-9c35-4912fd72082d.png)

training이 진행되면서 loss와 accuracy를 시각적으로 그래프로 확인하기위해 설정한다.

![image](https://user-images.githubusercontent.com/81463668/113806997-4c8f0380-979e-11eb-9482-bf95d50ce00a.png)

loss는 줄어들고 accuracy는 높아지는 것을 확인할 수 있는데 일정 epoch수에 다다르면 loss와 accuracy 둘다 크게 변하지 않는 모습을 보인다.

![image](https://user-images.githubusercontent.com/81463668/113807005-50bb2100-979e-11eb-911a-69771ff2d836.png)

이번에는 test를 불러와서 전처리를 해보도록하겠다.

![image](https://user-images.githubusercontent.com/81463668/113807014-557fd500-979e-11eb-854b-2912ab9b03d3.png)
![image](https://user-images.githubusercontent.com/81463668/113807022-57e22f00-979e-11eb-8a52-b5cfbea34ec2.png)
학습된 model에 test를 넣어서 Y값과 비교전에 예측값을 받아온다.

![image](https://user-images.githubusercontent.com/81463668/113807039-603a6a00-979e-11eb-8282-d9b687ec813b.png)

예측값은 확률처럼 0에서 1값을 가지는데 둘 중에 더 높은 값을 1로 만들고 아닌 값은 0으로 만드는 one-hot encoding을 실행한다.

![image](https://user-images.githubusercontent.com/81463668/113807051-63cdf100-979e-11eb-9db3-e9ea929b5de0.png)
![image](https://user-images.githubusercontent.com/81463668/113807059-6597b480-979e-11eb-84f1-17a3ec7d6f1f.png)

test_df의 category를 우리가 분류할 dog과 cat으로 변경시켜주는 작업이다.


![image](https://user-images.githubusercontent.com/81463668/113807070-6a5c6880-979e-11eb-8d39-ed9976b1e2fc.png)

이제 모든 학습이 종료되고 model에 test값을 넣어 예측값도 가져왔으니 matplot으로 이미지를 가져와서 그 이미지가 제대로 classification을 하는지 확인할 차례이다.
test set의 첫 번째 18개의 이미지를 갖고와서 정해진 크기로 만들고 이미지를 나열한 후 이미지에 대한 몇 번째 image인지 그리고 분류를 0 또는 1중에 어떤 것으로 분류했는지 나타내도록 했다.

![image](https://user-images.githubusercontent.com/81463668/113807087-6f211c80-979e-11eb-94f6-644e14d9d623.png)

대부분의 강아지와 고양이 이미지를 잘 분류해낸 결과를 볼 수 있는데 간혹 이미지에서 잘못 분류한 이미지들도 포함되어있었다. 이 부분이 loss가 아주 작지못한 부분이라고 생각할 수 있겠다.














