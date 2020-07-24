---
title: Semi-Supervised Classification with Graph Convolutional Networks
date: 2020-07-23
description: Published as a conference paper at ICLR 2017
categories:
    - Graph Convolutional Network
    - Semi-Supervised Learning
image: 

---

![슬라이드2](https://user-images.githubusercontent.com/33539468/88361872-80afc900-cdb5-11ea-9085-4d55d7c80752.PNG)


제목에서 제시된 두 가지 키워드를 먼저 살펴보겠습니다. 
Graph Convolutional Network는 기존에 이미지에 적용하던 convolution 연산을 그래프에 적용하여, CNN에서 filter의 weight sharing 작용과 유사하게 그래프에서도 같은 weight값이 여러 노드에 적용되어 노드의 분류 문제를 수행하는 모델입니다. 

이 GCN을 기반으로 semi-supervised learning을 하면 그래프의 몇몇 노드에 주어진 레이블을 활용하여 나머지 노드의 레이블을 예측할 수 있습니다.

그림처럼 일부 사람의 연구분야를 알고 있을 때 이 사람과 연결된 다른 사람들의 연구 분야를 예측하는 경우 본 논문에서 제시하는 GCN기반의 semi-supervised learning 방법을 적용할 수 있습니다.


<!-- ## Background -->

![슬라이드3](https://user-images.githubusercontent.com/33539468/88361874-81e0f600-cdb5-11ea-8d8d-19e454864969.PNG)

이전의 graph기반의 semi-supervised learning에서는 loss계산 시 label이 있는 노드에 대한 classification loss인 L0과 graph laplacian regularization term이 같이 작용했습니다. L0뿐만아니라 Lreg또한 최소가 되는 방향으로 학습하다 보니 연결된 노드가 유사한 representatio을 가질 수 밖에 없었습니다. 하지만 이는 유사도 이외에 추가적인 정보를 표현하지 못하기 때문에 모델의 능력을 제한합니다. 

본 논문에서는 regularization term을 사용하지 않고 인접행렬을 통해 그래프 구조를 직접적으로 neural network model에 입력합니다. 어떤 노드들이 연결되있는지에 대한 정보를 직접적으로 이용하고, supervised loss인 L0로부터 기울기 정보를 연결된 노드에 분산시켜 노드들의 representation을 학습시킵니다. 


<!-- ## Main Idea -->

![슬라이드4](https://user-images.githubusercontent.com/33539468/88361875-83122300-cdb5-11ea-8683-fd20027c7251.PNG)

논문에서 전반적인 학습 과정을 시각화하여 보여준 것입니다. 
graph G = (V, E)의 인접행렬 A와 각 노드의 feature X가 주어졌을 때, 
이를 활용하여 분류 문제를 해결하는 graph-based NN model f(X, A)를 어떻게 학습하는지에 대한 방법을 제시합니다. 

그림에서 node feature와 locality 등의 정보가 색깔이 있는 선으로 표현되어 있고 이것들이 hidden layer를 거치면서 주변 노드들에 전달이 되어 최종적으로 노드가 특정 레이블 y로 분류되는 것을 볼 수 있습니다. 이러한 정보를 포함한 hidden layer를 잘 학습하는 것이 학습의 목적이라고 볼 수 있습니다. 


<!-- ## Graph Convolutional Network -->

![슬라이드5](https://user-images.githubusercontent.com/33539468/88361876-84435000-cdb5-11ea-8c83-6cd78d53bdf2.PNG)


방금 시각화했던 과정을 수식으로 정리해보겠습니다. 
H: 노드의 초기 feature로는 citation network에서는 각 문서의 bag-of-words[^1]  vector를 사용했고, knowledge graph data에서는 entity와 entity를 연결하는 relation이 있을 때 relatio에 대한 one hot vector를 사용했습니다.

W: 이 식을 행렬연산을 해보면 weight의 각각의 columln이 하나의 filter로 볼 수 있는데, weight의 각 column이 모든 노드에 똑같이 적용되기 때문에 CNN에서 filter로 weight sharing을 하듯이 마찬가지로 그래프에서 weight sharing을 할 수 있습니다.

따라서 GCN은 각 노드마다 자기자신을 포함한 이웃 노드들의 feature의 가중합으로 표현되는 것을 알 수 있습니다. 

![슬라이드6](https://user-images.githubusercontent.com/33539468/88361877-860d1380-cdb5-11ea-8e96-401ccab12663.PNG)

GCN은 크게 spectral GCN과 spatial GCN이 있습니다.
spectral GCN에서는 그래프의 laplacian 행렬의 고윳값분해를 수행합니다. 고윳값 분해는 그래프 sub-group이나 cluster와 같은 구조를 표현할 수 있고, 라플라시안 행렬을 통해 각 노드의 degree를 포함한 그래프 전체 구조를 인코딩함으로써 노드 간 신호가 어떻게 전달되는지 이해할 수 있습니다.
하지만 그래프의 모든 노드를 동시에 입력하기를 요구하기 때문에 시간복잡도가 N^2이 되어 크기가 큰 그래프일수록 cost가 너무 증가한다는 단점이 있습니다.

반면 spatial GCN은 이웃 노드들에 대해서만 convolution을 수행하기 때문에 그래프 전체가 동시에 처리될 필요가 없어서 병렬적으로 계산을 수행할 수 있고 계산량을 절감할 수 있습니다. 최근에는 이러한 이점 때문에 spatial GCN을 많이 사용한다고 합니다.

본 논문에서는 Chebyshev 다항식을 통해 전체 그래프가 아닌, K step까지의 이웃 노드만 고려해서 연산량이 감소된 spectral-based GCN을 사용합니다. 


<!-- ## Spectral Graph Convolution -->

![슬라이드7](https://user-images.githubusercontent.com/33539468/88361878-860d1380-cdb5-11ea-8e46-27f6079f711f.PNG)

Chebyshev polynomial을 적용하기 전과 후의 spectral graph convolution 식입니다. signal x는 노드 벡터이고, Laplacian matrix의 고윳값에 대한 필터입니다. chebyshev를 적용하기 전 식에서 오른쪽 수식은 필터와 입력값을 spectral convolution한 것을 고유값 분해한 것입니다. 이 때 그래프의 모든 노드를 고려했기 때문에 eigenvector matrix인 U는 N^2의 시간복잡도를 요구합니다. 

Cheyshev polynomial을 도입한 식에서는 필터 g를 k차 이웃 노드까지까지만 제한하여 연산량을 edge 개수에 선형적인 O(E)로 감소시킵니다. 이렇게 K를 특정 값으로 고정시킨, 연산량이 고정된 레이어를 쌓아서 깊은 모델을 만들 수 있습니다. 


<!-- ## Renormalization trick of Laplacian -->

![슬라이드8](https://user-images.githubusercontent.com/33539468/88361880-873e4080-cdb5-11ea-8078-09a361ad3edb.PNG)

논문에서는 K=1, 최대 고윳값 = 2로 가정하고, 가장 가까웃 이웃과 두번째로 가까운 이웃 노드에 같은 weight를 줘서 GCN이 overfitting하지 않게끔 파라미터 수를 줄여줍니다. k = 1은 Chebyshev Polynomial을 0, 1번째 항으로 제한함으로써 spectral graph convolution의 1차 근사를 의미합니다.

이 세팅을 기반으로 filter와 노드를 convolution했을 때, 인접행렬의 정규화를 통해 각 노드별 edge의 갯수에 관계없이 모든 노드들을 잘 학습할 수 있습니다. 
renormalization trick을 적용하기 전의 식은 normalized laplacian 식입니다. 이 식에서는 각 레이어마다 identity matrix를 포함하게 되는데, 이 identity matrix 때문에 layer를 거듭할수록 convolution 값이 커지게 됩니다. 


eigenvalue가 1보다 작으면 gradient vanishing이 발생하고 1보다 크면 gradient exploding이 발생하는 데, 논문에서는 eigenvalue의 최댓값이 2라서 eigenvalue가 [0, 2]범위를 가지기 때문에 gradient exploding/vanishing 문제가 발생한다고 하였습니다. renormalization을 통해 identity matrix를 소거하여 이 문제를 해결합니다.


<!-- ## Semi-Supervised Learning  -->

![슬라이드9](https://user-images.githubusercontent.com/33539468/88361882-87d6d700-cdb5-11ea-9f65-68857eb445bc.PNG)

본 논문에서는 위 식과 같이 2-layer GCN으로 진행되었습니다.  비선형 함수로 첫 번째 layer에서는 ReLU, 두 번째 layer에서는 classification을 위해 Softmax를 이용했습니다.

2 layer 이기 때문에 첫번째 weight는 input to hidden, 두 번째 weight는 hidden to output weight matrix로 작용합니다. 

분류 문제에서는 분류 오차에 대한 정확도를 고려할 필요가 있기 때문에 Mean squared error보다 cross entropy를 사용합니다. 본 논문도 분류 문제를 다루기 때문에 cross entropy를 사용해 loss를 계산합니다. 



<!-- ## Experiment -->

![슬라이드10](https://user-images.githubusercontent.com/33539468/88361883-886f6d80-cdb5-11ea-8d44-cf7cfc632242.PNG)

실험은 크게 citation network dataset과 knowledge graph dataset, 그리고 random graph로 나뉩니다. citation network에서 노드는 document이고 edge는 citation link입니다. feature는 각 document에 대한 bag-of-words vector이고 citation link는 undirected edge로 구성됩니다. 

knowledge graph 데이터는 특정 entity들과 그 사이의 관계(relation)를 갖고 있습니다. citation network에서는 각 클래스별로 20개의 label 데이터를 사용하는 반면에 knowledge graph에서는 클래스 별로 label 데이터를 하나씩만 사용합니다. 

random graph는 epoch 별로 training 시간을 측정하기 위해서 다양한 크기의 그래프 데이터셋에 대해 저자가 실험한 것입니다. input feature matrix 로 노드 크기만큼의 identity matrix를 도입해서 feature가 없는 것으로 가정하여 실험합니다.



![슬라이드11](https://user-images.githubusercontent.com/33539468/88361884-8a393100-cdb5-11ea-85c3-3b31f8eee618.PNG)


training 시에 full batch gradient descent 방법을 사용했기 때문에 모든 training iteration마다 전체 데이터셋이 메모리에 있도록 했습니다. 논문 말미에 저자가 future work로 full batch 대신에 mini batch를 사용해서 크기가 큰 그래프에 대해서도 메모리 오버플로우 없이 GCN이 잘 작동할 필요가 있다고 언급했습니다. 

validation에는 500개의 labeled data를 사용했고, 오버피팅을 방지하기 위해 dropout과 L2 regularization에 대한 하이퍼파라미터를 변경하며 실험했습니다.

test에는 1000개의 labeled data를 사용했습니다.

<!-- ## Results -->

![슬라이드12](https://user-images.githubusercontent.com/33539468/88361885-8a393100-cdb5-11ea-95d5-4a822d3c76af.PNG)

기존의 최신(SOTA) 방법들과 같은 환경에서 실험 후 GCN의 정확도를 비교한 표입니다. random weight로 초기화한 후 100번의 결과를 평균하였을 때, 이전 classification 모델들보다 우수한 성능을 내었습니다. 

![슬라이드13](https://user-images.githubusercontent.com/33539468/88361886-8ad1c780-cdb5-11ea-9784-3c79c1eb8257.PNG)

마찬가지로 random weight로 초기화한 후 100번의 결과를 평균해서 propagation 방법을 다르게 하였을 때, K=1, 최대 고윳값 = 2로 가정하고 renormalization trick을 적용한 경우가 가장 정확도가 높았습니다. 

<!-- ## Future work -->

![슬라이드14](https://user-images.githubusercontent.com/33539468/88361889-8c9b8b00-cdb5-11ea-873d-4f024256402d.PNG)

full batch gradient descent를 쓰면 dataset 크기에 선형적으로 메모리가 소요되기 때문에, mini-batch로 cost를 줄일 필요가 있습니다. 단, K차 이웃 노드까지 계산에 필요하기 때문에 K번째 이웃노드들이 메모리에 동시에 있어야 합니다.

본 논문은 기본적으로 indirected graph에만 적용가능 합니다. NELL처럼 directed graph일경우 edge를 노드로 간주해 이분 그래프로 바꾸어 표현할 필요가 있습니다.

실험에서는 self-connecton과 이웃 노드간의 edge에 똑같은 가중치를 두어서 계산하였으나 데이터셋에 따라 self-connection에 가중치를 두고 학습할 필요가 있습니다. 이 때 이 가중치는 gradient descent를 통해 적절한 값으로 학습이 가능합니다.





## Reference
[1]<https://personal.utdallas.edu/~hkokel/articles/GraphConvolutionalNetwork.html>

[2] <https://www.slideshare.net/SEMINARGROOT/graph-convolutional-network-234828943>

[3] <https://towardsdatascience.com/an-introduction-to-graph-neural-networks-e23dc7bdfba5>

[4] <https://baekyeongmin.github.io/paper-review/gcn-review/>


[^1]: Bag of Words란 단어들의 순서는 전혀 고려하지 않고, 단어들의 출현 빈도(frequency)에만 집중하는 텍스트 데이터의 수치화 표현 방법입니다.