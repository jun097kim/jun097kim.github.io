---
layout: post
title: MG-VTON 논문 요약
math: true
---

## Abstract

임의의 포즈 하에서의 Virtual try-on system은 다양한 응용 가능성을 가지고 있지만, 많은 도전 과제가 있다.  
e.g. self-occlusions, 다양한 포즈에 따른 오배치, 다양한 옷의 질감의 표현

옷을 피팅하는 기존의 방법은 고정된 포즈로 디테일한 질감을 잃어버리고, 포즈의 다양성이 줄어드는 불완전한 성능을 보여주었다.

이 논문에서는 옷을 다양한 포즈의 사람 이미지에 합성할 수 있는 multi-pose 정보를 활용하는 virtual try-on-system을 처음으로 시도하였다.

사람 이미지, 원하는 옷 이미지, 원하는 포즈가 인풋으로 주어지면 MG-VTON은 새로운 사람 이미지를 생성할 수 있다.

MG-VTON은 **3단계**로 구성된다.

1. 타겟 이미지의 human parsing map은 원하는 포즈와 옷의 모양을 만족시키면서 합성된다.
2. Warp-GAN은 원하는 옷 형태를 합성된 human parsing map으로 와핑시킨다.
3. multi-pose composition masks를 활용한 refinement render가 디테일한 질감을 복구시킨다.

**\*human parsing:** 사람 이미지를 머리, 몸통, 팔과 다리와 같은 작은 의미있는 부분으로 나누는 것이다. (출처: Progressive Cognitive Human Parsing)

## 1. Introduction

포즈를 조작하면서 옷 이미지를 사람 이미지에 합성하는 것을 배우는 것은 virtual try-on, virtual reality(VR), 사람과 컴퓨터의 상호작용과 같은 많은 애플리케이션에 중요하다.

옷과 포즈 2가지를 조건으로 하는 사람 이미지를 합성하는 multi-stage 방법을 제안한다.

![cf07158a.png](/assets/mg-vton-results.png)

**첫 번째 행**은 인풋으로 사용된 옷 이미지이고, **첫 번째 열**에 인풋으로 사용된 사람 이미지, **나머지 열**에는 합성된 결과가 보인다.

사람 이미지, 원하는 옷, 원하는 포즈가 주어지면 원하는 옷과 사람의 모습을 보존하면서 포즈를 재구성한다.

### 기존 Virtual Try-on의 문제점

임의의 포즈로 옷을 합성하는 것은 쇼핑할 때 옷을 선택하는데에 도움을 준다. 하지만, virtual try-on을 위한 최근의 이미지 합성 접근법은 고정된 포즈에 집중되어 있고, 하의와 헤어과 같은 디테일을 잃어버린다.

이러한 방법은 이미지를 생성하기 위해 오로지 옷만 조건으로 하여 coarse-to-fine network를 사용한다. 그래서 human-parsing의 중요한 피쳐들을 잃어버리고, 특히 다양한 포즈를 조건으로 할 때 **흐릿한(blurry) 이미지**를 생성한다.

![MG-VTON Comparison](/assets/mg-vton-comparison.png)

위 그림에서 보듯이 상의를 교체하는 동안에 **하의**가 보존되지 않으며, 다른 포즈에서는 **사람의 머리**를 인식할 수 없다.

다른 관련된 연구들은 이 이슈를 해결하기 위해 일반적으로 **3D 측정 정보**를 활용한다. 그 이유는 3D 정보가 합성 결과를 생성하는데 도움이 되는 체형에 대한 충분한 정보를 가지고 있기 때문이다.

하지만 3D 모델을 만드는 데에는 전문 지식이 필요하고, 막대한 인건비가 필요하다. 이러한 비용과 복잡성으로 인해 실제 virtual try-on 시뮬레이션에서의 응용에 한계가 있다.

이 논문에서는 2D 이미지와 임의의 포즈를 조건으로 하는 virtual try-on을 연구했다. 이 인풋으로 사용되는 **사람 이미지**를 옷과 포즈가 다른 **같은 사람의 다른 이미지**로 매핑하는 함수을 학습하는 것을 목표로 한다.

fixed-pose virtual try-on이 광범위하게 연구되었지만, multi-pose virtual try-on은 덜 연구되었다.

게다가 외모, 옷, 포즈 간의 복잡한 상호 작용의 매핑을 모델링하지 않고 기존의 virtual try-on 방법으로 다른 포즈에 이미지를 합성하면 사용하면 **흐릿하고 인공적인 요소**들이 보일 때가 많다.

### MG-VTON 제안

위에서 언급한 문제들을 해결하기 위해 인풋 이미지에 원하는 옷을 피팅하고 포즈를 조작한 후 새로운 사람 이미지를 생성할 수 있는 Multi-pose Guided Virtual Try-on Network (MG-VTON)을 제안한다.

MG-VTON은 GAN을 기반으로 하는 multi-stage 프레임워크이다. 구체적으로는 **포즈와 옷 정보를 활용하는 human parsing 네트워크**를 설계하였다.

몸 형태, 얼굴 마스크, 머리 마스크, 원하는 옷, 타겟 포즈를 조건으로 하여 타겟 이미지로부터 **human parsing**을 추정한다. 이것은 신체 부위의 정확한 영역으로 합성을 유도할 수 있다.

원하는 옷을 사람에게 매끄럽게 피팅시키기 위해 **인풋으로 사용되는 옷 이미지의 마스크**와 합성된 human parsing에서 추출한 **합성된 옷의 마스크** 사이에 변환 파라미터를 추정하는 **geometric matching model**을 활용하여 옷 이미지를 와핑시킨다.

추가적으로 우리는 다양한 포즈와 옷으로 인해 생기는 오배치를 줄여주기 위해 **Warp-GAN**을 설계했다.

마지막으로 디테일한 질감을 복구시키고, 참조 포즈와 타겟 포즈 사이의 오배치로 인해 생기는 인공적인 요소를 줄여주는 multi-pose 구성 마스크를 활용하는 refinement network를 제시한다.

### MG-VTON 평가

우리의 모델을 증명하기 위해 다양한 옷 이미지와 같은 사람이 다양한 포즈로 있는 사람 이미지를 수집하여 MPV라는 새로운 데이터셋을 만들었다. 또한 테스트를 위해 DeepFashion 데이터셋에 대한 실험도 수행했다. 다른 논문에 나온 object evaluation protocol에 따라 Amazon Mechanical Turk (AMT) 플랫폼에서 인간 주관적 연구를 실행했다.
(GAN은 평가 방법이 마땅치 않아 이미지 생성이 잘 되었는지 Amazon에 외주를 주어 평가하는 방법이 있다.)

결과는 양적이나 질적으로 효과적인 성능과 디테일을 표현하는 고품질 이미지를 얻을 수 있음을 나타내었다.

주요 작업 내용은 다음과 같다:

- 다양한 포즈와 옷을 조작하여 사람 이미지를 재구성하는 것을 목표로 multi-pose를 조건으로 하는 virtual try-on을 제안하였다.
- 우리는 원하는 옷을 인풋으로 사용된 사람 이미지에 피팅시키고 포즈를 조작한 후 새로운 사람 이미지를 생성하는 Multi-pose Virtual Try-on 네트워크를 제안한다.

  1. 포즈와 옷 정보를 활용하는 **human parsing network**는 이미지 합성을 가이드하도록 설계되었다.
  2. **Warp-GAN**은 와핑된 피쳐를 사용하여 현실 이미지를 학습하는 것을 학습한다.
  3. **refinement 네트워크**는 디테일한 질감을 복구하는 것을 학습한다.
  4. 마스크 기반의 **geometric matching 네트워크**는 와핑된 옷의 생성된 이미지의 시각적 품질을 향상시킨다.

- multi-pose 정보를 활용하는 virtual try-on 작업을 위한 다양한 포즈와 옷을 다루는 새로운 데이터셋을 수집했다. 광범위한 실험은 양적이고 질적인 결과를 얻을 수 있음을 보여주었다.

## 3. MG-VTON

![MG-VTON Overview](/assets/mg-vton-overview.png)

MG-VTON 개요.  
Stage 1: 먼저 참조 이미지(reference image)를 3개의 바이너리 마스크(헤어, 얼굴, 몸통 마스크)로 분해한다. 그 다음 human parsing map을 예측하는 **conditional parsing 네트워크**의 인풋으로 사용하기 위해 타켓 옷과 포즈를 마스크들과 합친다.  
Stage 2: 다음으로 옷을 와핑한다. 참조 이미지에서 옷 영역을 삭제한 후 타겟 포즈와 synthesis parsing을 합쳐서 Warp-GAN을 통해 1차 합성 결과(coarse result)를 만든다.  
Stage 3: 마지막으로 와핑된 옷, 타겟 포즈, coarse result를 조건으로 하는 refinement render를 통해 1차 합성 결과를 정제한다.

옷과 포즈 모두를 조작함으로써 virtual try-on을 위한 새로운 사람 이미지를 합성하는 것을 학습하는 Multi-pose 정보 활용 Virtual Try-on 네트워크를 제안한다.

인풋으로 사람 이미지, 원하는 옷, 원하는 포즈가 주어지면 MG-VTON은 원하는 옷을 입고 포즈를 조작한 사람의 새로운 이미지를 생성하는 것을 목표로 한다.

coarse-to-fine 아이디어에서 영감을 받아 이 작업를 세 가지 하위 작업으로 나누는 outline-coarse-fine 전략을 채택했다.

### MG-VTON 개요

먼저 **pose estimator**를 이용하여 포즈를 추정한다. 그런 다음, 포즈를 18개의 히트맵으로 인코딩한다. 이 히트맵은 반경이 4픽셀인 원은 1로, 나머지는 0으로 채워진다.

**human parsesr**는 얼굴, 헤어, 몸체의 바이너리 마스크를 추출하는 20개의 레이블로 구성된 human segmentation maps을 예측할 때 사용된다.

VITON에 따라 몸체를 낮은 해상도로 (16 x 12)로 다운샘플링하고, 다시 원래 해상도로 리사이즈하여 다양한 몸의 형태로 인한 인공적인 요소를 줄여준다.

## 3.1 Conditional Parsing Learning

옷과 포즈를 조작하는 동안에 사람의 구조적 일관성을 유지하기 위해 우리는 옷 이미지, 포즈 히트맵, 추정된 몸의 형태, 얼굴 마스크, 헤어 마스크를 조건으로 하여 **포즈와 옷 정보를 활용하는 human parsing network**를 설계했다.

![MG-VTON Comparison](/assets/mg-vton-comparison.png)

위 그림에서 보듯이 기존 방식은 사람과 옷 이미지를 모델에 직접 넣어주기 때문에 사람의 특정 영역을 보존하는데 실패했다. (e.g. 바지 색상과 머리 스타일이 바뀌었다.)

여기서는 human parsing maps을 활용하여 이런 문제를 해결하였고, 이는 generator가 고품질 이미지를 생성하는데 도움이 된다.

### MG-VTON 네트워크 아키텍처

**이를 수식화하면 다음과 같다.**

인풋 사람 이미지 $I$, 인풋 옷 이미지를 $C$, 타겟 포즈 $P$가 주어진다면, 옷 $C$와 포즈 $P$라는 조건 하에서 human parsing map $S^{'}_{t}$를 예측하는 것을 학습한다고 해 보자.

![MG-VTON Network Architecture](/assets/mg-vton-network-architecture.png)

MG-VTON의 네트워크 아키텍처. (a)(b): conditional parsing 학습 모듈은 clothes-guided network로 구성되어 human parsing을 예측한다. (c\)(d): Warp-GAN은 포즈의 다양성으로 인한 오배치 문제로 인해 와핑 피쳐 전략을 사용하여 이미지를 생성하는 것을 학습한다. (e): refinement render 네트워크는 합성 이미지의 시각적 품질을 향상시키는 pose-guided composition mask를 합성시킨다. (f): geometric matching network는 몸 형태와 옷 마스크를 조건으로 하는 변환 매핑을 추정하는 것을 학습한다.

위 그림 (a)에서 볼 수 있듯이, 먼저 헤어 마스크 $M_{h}$, 얼굴 마스크 $M_{f}$, 몸 형태 $M_{b}$, 타겟 포즈 $P$를 추출한다. 그 다음 이것들과 옷 이미지를 연결하여 conditional parsing 네트워크의 인풋으로 넣어준다.

$S^{\'}\_{t}$의 inference는 사후 확률 $p(S^{\'}\_{t}\|(M_{h}, M_{f}, M_{b}, C, P))$을 최대화하는 것으로 수식화할 수 있다. 또한, 이 단계는 이미지 조작에 좋은 결과를 생성하는 conditional generative adversarial network (CGAN)을 기반으로 한다.

따라서 posterior probability는 다음과 같이 표현된다:

$$
p(S^{'}_{t}|(M_{h}, M_{f}, M_{b}, C, P)) = G(M_{h}, M_{f}, M_{b}, C, P)
$$

**ResNet**과 같은 네트워크를 generator $G$로 채택하여 conditional parsing model을 작성한다. 그대로 **pix2pixHD**의 discrimiator $D$를 채택했다. 성능을 더 향상시키기 위해 **L1 loss**를 적용하였는데, 이는 보다 부드러운 결과를 생성하는데 유리하다. **pixel-wise softmax loss**를 적용하여 generator가 고품질의 human parsing maps을 합성하도록 한다.

그러므로 conditional parsing 학습은 다음과 같이 수식화될 수 있다:

$min_{G} \max_{D} F(G, D)$  
$= E_{M,C,P \sim P_{data}}[log(1-D(G(M,C,P), M,C,P))]$  
$+ E_{S_t,M,C,P \sim p_{data}}[log(D(S_t,M,C,P))]$  
$+ E_{S_t,M,C,P \sim p_{data}}[||S_t - G(M,C,P)||]$  
$+ E_{S_t,M,C,P \sim p_{data}}[L_{parsing}(S_t,G(M,C,P))]$

여기서 M은 $M_{h}, M_{f}, M_{b}$을 묶은 것이다. loss $L_{parsing}$은 pixel-wise softmax loss를 나타낸다. $S_{t}$는 human parsing의 참 값을 나타낸다. $p_{data}$는 실제 데이터 분포를 나타낸다.

## 3.2 Warp-GAN

픽셀의 오배치로 인해 흐릿한 결과가 생성되기 때문에, deep Warping Generative Adversarial Network (Warp-GAN)을 도입하여 **원하는 옷 모양**을 합성된 human parsing map에 와핑시킨다.

다른 논문 deformableGANs과 [1]과는 달리, affine과 TPS (Thin-Plate Spline)을 사용하여 bottleneck 레이어에서 feature map을 와핑시킨다.

다른 논문 [23]의 일반화 능력 덕분에 pre-trained 모델을 사용하여 **참조 parsing과 합성된 parsing** 사이의 변환 매핑을 추정한다.

그런 다음 이 변환 매핑을 사용하여 **w/o(without) clothes 레퍼런스 이미지**를 와핑시킨다.

![MG-VTON Network Architecture](/assets/mg-vton-network-architecture.png)

위 그림 (c\)와 (d)에서 볼 수 있듯이, 제안된 deep warping 네트워크는 Warp-GAN generator $G_{warp}$와 Warp-GAN discriminator $D_{warp}$로 구성된다. 3.4절에서 설명한 대로 **geometric matching 모듈**을 사용하여 옷 이미지를 와핑시킨다.

**이를 수식화하면 다음과 같다.**

와핑된 옷 이미지 $C_{w}$, w/o clothes 참조 이미지 $I_{w/o\_clothes}$, 타겟 포즈 $P$, 합성된 human parsing $S^{'}_{t}$를 Warp-GAN generator의 인풋으로 넣어서 결과를 합성한다.

$$
\hat{I} = G_{warp} (C_{w}, I_{w/o\_clothes}, P, S^{'}_{t})
$$

**perceptual loss**를 적용하여 pre-trained 모델의 고차원 피쳐들 간의 거리를 측정한다. 이것은 generator가 고품질의 사실감 있는 이미지를 합성하도록 한다.

perceptual loss를 수식화하면 다음과 같다:

$$
L_{perceptual}(\hat I, I) = \sum\limits_{i=0}^n \alpha_{i}||\phi_{i}(\hat i) - \phi_{i}(I)||_{1}
$$

**\*perceptual loss:** 똑같은 feature matching loss인데, pre-trained 모델에 적용한 것이라고 보면 되겠다.

여기서 $\phi_{i}$는 pre-trained 네트워크의 i번째 레이어 feature map을 나타낸다. 우리는 pre-trained VGG19를 $\phi$라고 하고 이미지 간의 perceptual loss를 나타내기 위해 $\phi$의 마지막 5개 레이어의 L1 norms을 가중치 합산한다.

$\alpha_{i}$는 각 계층의 loss에 대한 **가중치**를 나타낸다.

pix2pixHD에 따르면, discriminator의 서로 다른 레이어에서 다른 규모의 feature map으로 인해 이미지 합성의 성능을 향상시키므로, **feature loss**를 도입하고 다음과 같이 수식화한다:

$$
L_{feature}(\hat I, I) = \sum\limits_{i=0}^n \gamma_{i}||F_{i}(\hat i) - F_{i}(I)||_{1}
$$

여기서 $F_{i}(I)$는 학습된 $D_{warp}$의 i번째 레이어 feature map을 나타낸다. $\gamma_{i}$은 해당 계층에 대한 L1 loss의 가중치를 나타낸다.

**\*feature (matching) loss:** geneator의 학습 성능을 향상시키기 위해 추가하는 loss

또한, 성능을 향상시키기 위해 adversarial loss $L_{adv}$, L1 loss $L_{1}$을 적용한다.

$G_{warp}$의 loss로써 가중치 합 loss를 설계했다. 이것은 $G_{warp}$가 현실적이고 자연스러운 이미지를 합성하도록 한다.

여기서 $\lambda_{i}$는 각각 해당하는 loss의 가중치를 나타낸다.

## 3.3 Refinement render

coarse 단계에서 사람의 식별 정보와 모습은 유지될 수 있지만, 옷 이미지의 복잡성으로 인해 **디테일한 질감**을 잃어버린다.

와핑된 옷을 사람에게 직접 붙여 넣으면 인공적인 요소가 생성될 수 있다. 와핑된 옷 이미지와 1차 합성 결과 사이의 composition mask를 학습하면 포즈의 다양성으로 인해 인공적인 요소가 생성된다.

위의 문제를 해결하기 위해 multi-pose composition 마스크를 사용하여 디테일한 실감과 일부 인공적인 요소를 제거하는 **refinement render**를 제시한다.

**이를 수식화하면 다음과 같다.**

우리는 $C_w$를 geometric matching 모듈에 의해 얻어진 와핑된 옷 이미지로, $\hat I_c$는 Warp-GAN에 의해 생성된 coarse result로, $P$는 타겟 포즈 히트맵으로, $G_p$는 refinement render의 generator로 정의한다.

![MG-VTON Network Architecture](/assets/mg-vton-network-architecture.png)

위 그림 (e)에서 볼 수 있듯이, Refinement Render Generator는 $C_w$, $\hat I_c$, $P$를 인풋으로 하고 $G_p$는 multi-pose composition mask를 예측하여 렌더링된 결과를 합성한다.

$$
\hat I_p = G_p(C_w, \hat I, P) \odot C_w + (1 - G_p(C_w, \hat I, P)) \odot \hat I
$$

여기서 $\odot$은 원소 단위의 **행렬 곱**을 나타낸다. 우리는 성능을 향상시키기 위해 **perceptual loss**를 채택했다.

$G_p$의 목적 함수가 다음과 같이 쓰여질 수 있다:

$$
L_p = \mu_1 L_{perceptual} (\hat I_p, I) + \mu_2||1 - G_p(C_w, \hat I_c, P)||_1
$$

$\mu_1$은 perceptual loss의 가중치를 나타내고 $\mu_2$는 mask loss의 가중치를 나타낸다.

## 3.4 Geometric matching learning

피쳐 추출 레이어, 피쳐 matching 레이어, 변환 파라미터 추정 레이어를 포함하는 deconvolutional neural network를 채택했다.

![MG-VTON Network Architecture](/assets/mg-vton-network-architecture.png)

위 그림 (f)에서 볼 수 있듯이, 옷 이미지의 마스크와 몸 형태의 마스크가 먼저 **피쳐 추출 레이어**의 인풋으로 전달된다.
그런 다음, matching layers를 사용하여 두 마스크 사이의 상관관계 맵인 **correlation map**을 예측한다.
마지막으로, 회귀 네트워크를 적용하여 correlation map을 기반으로 **TPS (Thin-Plate Spline) 변환 파라미터**를 추정한다.

**이를 수식화하면 다음과 같다.**

conditional parsing 학습 단계 이후에 인풋 이미지로 옷 $C$와 마스크 $C_{mask}$가 주어지면, 합성된 human parsing으로부터 몸 형태 $M_b$와 합성된 옷 마스크 $\hat C_{mask}$를 얻는다.

이 하위 작업은 인풋 옷 이미지 $C$를 와핑하기 위해 파라미터 $\theta$를 가진 변환 매핑 함수 $T$를 학습하는 것을 목표로 한다.

합성된 옷은 보이지 않지만 **합성된 옷의 마스크**를 가지고 있기 때문에, 몸 형태 ${M_b}$에 맞게 원래의 옷 마스크 $C_{mask}$에서 합성된 옷 마스크 $\hat C_{mask}$를 매핑하는 것을 학습한다.

따라서 geometric matching function의 목적 함수는 다음과 같이 수식화될 수 있다:

$$
L_{geo\_matching}(\theta) = ||T_{\theta}(C_{mask} - \hat C_{mask})||_1
$$

그러므로 와핑된 옷 $C_w$는 $C_w = T_{\theta}(C)$로 수식화될 수 있다. 이는 오배치 및 compoistion 마스크 학습 문제를 해결하는데 도움을 준다.

## 4. Experiments

이 장에서는 먼저 다른 방법과 시각적으로 비교한 다음 그 결과를 정량적으로 논한다.

## 4.3 Implementation Details

**환경.** ADAM optimizer를 사용하여 conditional parsing 네트워크, Warp-GAN, refinement render, geometric matching network를 각각 200, 15, 5, 35 epochs만큼 배치 사이즈는 40, learning rate는 0.0002, $\beta_1 = 0.5$, $\beta2 = 0.999$로 학습했다.

2개의 NVIDIA Titan XP GPU, Ubuntu 14.04에서 Pytorch 플랫폼을 사용했다.

**아키텍처.** MG-VTON의 각 generator는 ResNet과 유사한 네트워크로 3개의 다운샘플링 레이어, 3개의 업샘플링 레이어, 9개의 residual 블록으로 구성되어 있다. 각 블록은 3x3 필터 커널이 있는 3개의 convolutional 레이어를 가지고 있으며, 이어서 batch-norm 레이어와 Relu 활성화 함수를 거친다. 필터 수는 64, 128, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256, 128, 64이다.

discriminator는 서로 다른 레이어로 다른 규모의 피쳐 맵을 다룰 수 있는 pix2pixHD와 동일한 아키텍처를 적용했다. 각각의 discriminator는 4x4 커널과 InstanceNorm, LeakyReLU 활성화 함수를 포함하는 4개의 다운샘플링 레이어를 포함한다.
