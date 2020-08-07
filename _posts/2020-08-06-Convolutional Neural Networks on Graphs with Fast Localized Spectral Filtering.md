---
title: Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering
date: 2020-08-06
description: Published as a conference paper at NIPS 2016
categories:
    - Graph Convolutional Network
image: 

---

![슬라이드1](https://user-images.githubusercontent.com/33539468/89627597-1713dc80-d8d6-11ea-8b08-db197c9fccc9.PNG)

Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering[^1] 논문을 정리, 요약한 글입니다. 제가 잘못 이해한 부분이 있거나 오타, 오역 등의 의견은 댓글과 메일로 남겨주시면 감사하겠습니다.^^

![슬라이드2](https://user-images.githubusercontent.com/33539468/89627601-17ac7300-d8d6-11ea-952e-6a701d7dd57c.PNG)

본 논문은 CNN을 저차원 regular grid에서 고차원 irregular domain에 일반화하고자 합니다. 오른쪽 그림의 윗부분에 해당하는 저차원, regular grid 데이터는 이미지, 비디오, speech를 예시로 들 수 있습니다. 아래 그림에 해당하는 고차원 데이터에는 social network, 유전자와 같은 생체 데이터, telecommunication network에서 로그데이터, 등이 있습니다. 이러한 불규칙적이고 non-euclidean domain의 데이터를 그래프로 표현 가능하고, spectral graph theory로 그래프의 기하학적인 구조를 수식화 가능합니다. 따라서 spectral graph theory 기반의 그래프 도메인에서 CNN을 수식화하여 표현할 수 있습니다. 




![슬라이드3](https://user-images.githubusercontent.com/33539468/89627603-18450980-d8d6-11ea-8927-67fa224fed5e.PNG)

논문 제목에 제시된 두 키워드를 통해 구체적으로 본 논문에서 무엇을 해결하고자 하는지 볼 수 있습니다. 
제목에서 제시된 localized와 fast 키워드에서 나타나듯이, 논문에서는 저차원 regular grid에서 그래프로 CNN을 일반화할 때 발생가능한 localization 문제를 해결하면서 동시에 계산 복잡도를 감소시켜 CNN을 그래프에 효율적으로 적용하는 방법을 제시합니다. 
논문 introduction에 이 localized filter와 low computatonal complexity를 포함해 총 5개의 contribution을 명시해 주었는데, 저는 제목에 포함된 이 2가지 contribution이 가장 중요한 요소라고 생각되어 이 두개를 먼저 짚고 나머지 3개를 이어서 설명하도록 하겠습니다. 

classical cnn은 large scale, 고차원 데이터셋에서 local structure를 학습하여 의미있는 패턴을 잘 찾아내는 것으로 알려져 있습니다. 이러한 local feature들은 filter로 표현되는데, 필터는 translation-invariant이기 때문에 필터의 위치나 인풋 데이터 사이즈에 관계없이 동일한 feature를 찾아낼 수 있습니다. 그런데 이 classical cnn을 그래프에 일반화할 때, convolution과 pooling 연산자가 regular grid에만 정의되어 있기 때문에 CNN을 graph에 바로 일반화할 수가 없습니다. 따라서 localized graph filter를 정의할 필요가 있고, 논문에서는 K차까지 localized된 라플라시안 다항식을 도입해 이 문제를 해결합니다.

두 번째로 classical CNN과 같은 complexity를 가지도록 learning complexity와 evaluation complexity를 모두 감소시켰습니다. n은 노드의 개수, k는 다항식의 k까지의 항을 의미합니다. learning complexity는 localization 전에 계산시 모든 노드의 개수를 요구했으나 localization 후에 중심 노드로부터 k hop까지의 노드만 요구함을 의미합니다. evaluation complexity는 n^2에서 k와 e에 대한 선형시간으로 개선되었는데 이 부분은 Spectral filtering of graph signals 파트에서 자세히 다루도록 하겠습니다. 


![슬라이드4](https://user-images.githubusercontent.com/33539468/89627606-18450980-d8d6-11ea-9a02-006b93f99aa9.PNG)

세 번째로 그래프 상에서 CNN의 spectral 구현을 수식화 할 때 기존에 존재하던 Graph Signal Processing 방법을 이용했습니다.

네 번째로 노드들을 binary tree 구조로 재정렬한 후에, 1D Signal을 pooling하는 것과 유사하게 pooling하여 GPU의 효율을 향상시킵니다.

마지막으로 MNIST와 text document데이터인 20-NEWS데이터에 대해 실험하고, 실험을 통해 본 논문에서 제시한 구조의 유용성, 계산석 효율성, 이전 spectral graph CNN들보다 accuracy와 complexity 측면에서 우수함을 보여줍니다. 



![슬라이드5](https://user-images.githubusercontent.com/33539468/89627608-18dda000-d8d6-11ea-9a10-d0f4311d2c96.PNG)

cnn을 그래프에 적용하는 전체적인 과정을 담은 그림입니다. bag of words와 같은 인풋 그래프 데이터를 받고, feature extraction, classification 후 label 등의 결과를 출력하는 전체 과정은 일반적인 cnn의 작용과정과 같습니다. 논문에서는 그림 하단 feature extraction 단계에서 CNN을 그래프에 일반화하기 위해 필요한 3가지 step에 주목하였습니다. 

1) 그래프 상에서 localized convolution filter의 설계<br />
2) 비슷한 의미를 가지는 vertex들을 모으는 그래프 압축<br />
3) graph pooling operation<br />

이 3가지 step에 대해서 자세히 다루도록 하겠습니다.

![슬라이드6](https://user-images.githubusercontent.com/33539468/89627609-19763680-d8d6-11ea-94c2-a3b85d52f1ea.PNG)

graph fourier transform은 undirected이고 connected graph G에서 Laplacian 연산자를 이용해 spectral graph 분석을 수행합니다. 
W는 노드 간의 연결을 표현하는 인접행렬이고 edge에 부여된 weight를 표현할 수 있습니다.

orthonormal eigenvector는 graph fourier mode라고도 불립니다.

signal x는 각각의 vertex를 real number인 x_i로 매핑하고, 이 실수 signal은 eigenvector들과 내적되어 graph fourier transform이 됩니다. 이 transform은 spectral filtering 연산을 가능하게 합니다.



![슬라이드7](https://user-images.githubusercontent.com/33539468/89627611-1a0ecd00-d8d6-11ea-8acc-6c841ef9739c.PNG)

먼저 localization을 수행하기 전의 spectral filtering을 살펴보겠습니다. classical cnn에서는 이미지 상에서 필터가 translation되면서 convolution이 수행되는데, graph의 vertex domain에서는 translation을 표현할 수 없으므로, fourier domain에서 convolution operator *g가 정의됩니다. 기호 ⊙(odot)는 Hadamard product인데, 이것은 같은 크기의 두 행렬의 각 성분을 곱하는 연산입니다. 저는 위 식에서 y를 필터로 생각해서 signal x와 필터 y가 U^Tx, U^Ty 형태로 graph fourier transform된 후에, pointwise product 후 다시 inverse graph fourier transform된 것이라고 해석했습니다. discrete fourier transform의 pointwise product가 convolution에 대응되는 것과 유사한 개념이라고 생각했습니다. 

튜닝해야할 파라미터가 없는 non-parametric filter인 𝑔_𝜃로 x를 필터링하여 filter된 신호 y를 출력합니다. 𝑔_𝜃의 세타는 fourier coefficient들의 벡터로 주기신호의 주파수 특성을 나타내주는 값입니다. 라플라시안 행렬을 eigenvalue decomposition한 후 도출되는 람다는 eigenvalue vector들이 모인 diagonal frequency matrix입니다. 따라서 frequency matrix에 대한 필터인 𝑔_𝜃 (Λ)를 fourier coefficient에 대한 대각행렬로 표현하였습니다. 이 필터는 모든 노드에 대해 적용되는 것이기 때문에 learning complexity는 data 차원의 수와 같고, computational complexity는 graph fourier basis의 곱연산으로 인해 n^2이 됩니다. 


![슬라이드8](https://user-images.githubusercontent.com/33539468/89627612-1a0ecd00-d8d6-11ea-8efa-43ab75c63b73.PNG)

non-parametric filter는 localization이 불가능하고 learning complexity가 O(N)인 2가지 한계점이 있었습니다. 논문에서는 이 두가지 문제를 chebyshev polynomial T_k(x)을 사용해 learning complexity와 localization 문제를 해결하였습니다.
𝑔_𝜃 (𝐿)을 sparse graph의 laplacian에 대한 다항식으로 표현해서 recursive하게 K번째 항까지 chebyshev 다항식이 계산되도록 합니다. 먼저 라플라시안을 scaling한 𝐿 ̃에 대한 k차 chebyshev 다항식 𝑇_𝑘 (𝐿 ̃ )을 구한 후, 𝑇_𝑘 (𝐿 ̃ )𝑥를 𝑥 ̅_𝑘 로 정의합니다. 정의된 변수를 사용하여 y가 𝑥 ̅에 대한 n*K 행렬과 세타에 대한 K벡터의 곱으로 표현됩니다. sparse graph는 에지 개수가 노드 개수에 선형적이므로 계산복잡도가 K|E|가 됩니다. 

제시하는 방법은 fourier basis matrix가 필요없으므로 O(𝑁^2) 행렬의 basis를 저장하고 계산량이 큰 고윳값분해를 수행할 필요가 없습니다. 이 방법은 lEl 크기의 none-zero 값을 가지는 sparse한 Laplacian 행렬만 저장하면 됩니다.


![슬라이드9](https://user-images.githubusercontent.com/33539468/89627613-1aa76380-d8d6-11ea-919c-cea83e2430aa.PNG)

feature extraction 단계에서, pooling을 하기 전에 비슷한 노드끼리 clustering 되어 있어야 의미있는 sub-sampling이 적용됩니다. 논문에서는 여러 level에 걸쳐 점점 더 압축된 그래프를 추출하는 multilevel clustering algorithm을 사용합니다. 각 레벨마다 그래프의 크기를 반으로 압축하는데 이 작업은 ‘Graclus’라는 clustering software를 사용합니다. 
Graclus는 마킹되지 않은 vertex i를 선택한 후 i의 마킹되지 않은 이웃노드 중 local normalized cut을 최대화하는 노드 j를 선택한 후 두 노드를 마킹합니다. normalized cut 식에서 𝑊_𝑖𝑗 는 노드 i와 j의 유사성을 의미합니다. 따라서 normalized cut을 최대화하는 이웃노드를 마킹하는 것은 가장 유사한 두 이웃노드끼리 그룹화해서 그래프를 1/2로 압축하는 것입니다. 

그림은 size 4 pooling의 예시입니다. Graclus는 그래프 크기를 1/2로 줄이기 때문에 size 4 pooling은 size 2 pooling을 2번하는 것으로 표현됩니다. 그래프 pooling을 1D pooling만큼 효율적으로 수행하기 위해 크게 2가지 단계로 진행됩니다. 1)먼저 balanced binary tree를 생성하고, 2) vertex를 재정렬하는 것입니다. pooling 시 노드는 singleton, fake regular 노드로 3종류가 있습니다. 초기 그래프를 g0은 파란색 fake노드를 제외한 나머지 노드 8개로 구성된 그래프입니다. 압축 후에 Graclus가 g0 압축 후에 마찬가지로 fake노드를 제외한 5개의 노드로 구성된 g1을 출력하고, g1 압축 후에 g2를 출력하였다고 합시다. 모든 노드가 2개의 children을 가지기 위해서 가장 압축된 level 2부터 level 0까지 차례대로 노드가 3개, 6개, 12개가 되어야 합니다. 
낮은 레벨에서부터 Graclus 압축 후에 match가 안 된 노드들은 빨간색 singleton이 되는데, 가상의 fake노드가 singleton과 짝을 이루어 children 쌍이 됩니다. level 2와 1의 연결의 보면 level 1의 0번 노드가 singleton이라 fake node 1이 추가된 것을 볼 수 있습니다. 
이렇게 가장 압축된 level2에서부터 level 0 순서대로 차례대로 정렬된 후에 level 0의 노드들에 1D pooling을 적용할 수 있습니다. 

fake node는 항상 2개의 fake node를 children으로 가지고 regular node와 singleton은 2개의 regular node를 갖거나 1개의 singleton과 1개의 fake node를 children으로 가집니다. fake node는 neutral value로 세팅하는데, 논문에서는 ReLU activation과 max pooling을 할 때 0을 사용했습니다.



![슬라이드10](https://user-images.githubusercontent.com/33539468/89627615-1aa76380-d8d6-11ea-9185-6f7913e5db23.PNG)

실험은 MNIST 이미지 데이터와 20NEWS라는 text data에 대해 진행되었습니다. 먼저 MNIST 실험에서는 각 노드는 이미지의 픽셀 하나에 해당합니다. 노드(픽셀) 간 유사성을 표현하는 graph weight matrix가 입력으로 사용되었습니다. classical CNN과 graph CNN을 구조를 똑같이 설계한 후 실험했을 때 정확도는 0.19 차이가 났습니다. 성능이 비슷하지만 gap이 존재하는데, 이 gap은 spectral filter의 isotropic 성질 때문입니다. 2D grid의 픽셀은 up, down, left, right와 같은 방향성을 가지고 있지만 일반적인 그래프의 edge들은 방향성을 표현하지 않습니다. 또한 image augmentation시에 rotation이 사용되는데 그래프 모델은 invariance를 학습하기 때문에 rotation 인지하지 못합니다.



![슬라이드11](https://user-images.githubusercontent.com/33539468/89627618-1b3ffa00-d8d6-11ea-8fe0-5ca84edb2fac.PNG)


두 번째 실험은 20NEWS 데이터셋으로 text categorization을 한 것입니다. 

논문에서 제시한 graph CNN 모델이 multimodal naïve bayes 분류기보다는 성능이 낮지만, fully connected network와 비교했을 때 fcn보다 적은 파라미터 수로 더 좋은 성능을 내었습니다. 


![슬라이드12](https://user-images.githubusercontent.com/33539468/89627620-1bd89080-d8d6-11ea-96ba-994823399b3c.PNG)

conclusion은 논문에서 기여한 5가지 요소로 마무리됩니다. graph convolutioinal layer로부터 local인 stationary feature를 추출하였고, data의 dimensionality에 선형적인 계산복잡도를 가지는 모델을 설계하였습니다. 또, Graph Fourier basis를 사용하지 않고 local filter를 사용해 test accuracy가 향상되었습니다.


![슬라이드13](https://user-images.githubusercontent.com/33539468/89627622-1bd89080-d8d6-11ea-8755-a4e3713523ba.PNG)

future work로는 graph signal processing이 발전함에 따라서 제안된 framework를 강화하고, 데이터의 종류와 구조에 의존하지 않는 일반적인 모델을 개발할 필요가 있음을 언급하였습니다. 또한 k-nn에서 k를 설정하듯이 학습이 불가능한 graph의 변수들을 바꿔가며 세팅할 필요가 있습니다.







## Reference
[1] https://www.slideshare.net/soyeon1771/convolutional-neural-networks-on-graphs-with-fast-localized-spectral-filtering

[2] Advances in Deep Learning on Graphs by Michaël Defferrard



## Footnote

[^1]: <https://papers.nips.cc/paper/6081-convolutional-neural-networks-on-graphs-with-fast-localized-spectral-filtering>