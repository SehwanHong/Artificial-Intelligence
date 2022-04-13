# [Deep Learning for Face Anti-Spoofing: A Survey](https://arxiv.org/pdf/2106.14948.pdf)

# Introduction

Face Recognition은 Presentation Attack에 취약합니다. 이러한 공격의 종류들로는 인쇄된 매체들, 또는 디지털 영상, 화장, 그리고 3D 마스크 등이 있습니다. 안전한 Face Recognition system을 만들기 위해서 수많은 연구원들이 PAs를 막기 위해서 노력하고 있습니다.

![Publications in Face Anti-Spoofing](../Publication_in_FAS.png)

위의 그래프에서 확인 할 수 있다 싶이, 최근 몇년간 나온 논문의 수가 증가한 것을 확인 할 수 있습니다.

FAS의 초기단계에서는 Handcrafted Feature를 이용해서 Presentation Attack Detection(PAD)를 했습니다. 이러한 Traditional한 방식은 전무 human liveness를 알 수 있는 것을 기반으로 만들어 졌습니다. 몇가지를 나열해보면, 눈 깜박임, 얼굴-머리 움직임(끄덕임, 미소), 시선 추적, 그리고 remote physiological signals가 있습니다. 이러한 정보들은 영상을 통해서 확인이 가능합니다. 그렇기 때문에 실제로 상업용으로 사용하기에는 약간의 무리가 있습니다. 또한 이러한 방식은 영상을 사용하는 것으로 무력화가 됩니다.

Classical Handcrafted Descriptors (LBP, SIFT, SURF, HOG, and DoG)는 다양한 색 영역(RGB, HSV, and YCbCr)에서 효율적인 spoofing pattern을 추출합니다.

몇몇 hybrid (handcrafted + Deep Learning)와 end-to-end deep learning은 Static 또는 dynamic한 PAD를 위해서 사용됩니다. 대부분의 FAS는 Binary Classification을 사용합니다. CNN with binary loss는 몇몇의 유용한 것을 확인할 수 있습니다. 예를 들어서 Screen bezel을 찾을 수는 있다. 제대로된 spooning pattern을 찾을 수는 없는 경우가 많다. 

Pixel-wise Supervision은 Fine-grain하고, context-aware한 신호를 찾는 다. 예를 들어서 Pseudo depth label, reflection map, binary mask label, 그리고 3D point Cloud map이 있다.

Generative deep FAS는 사용되었지만 아직은 연구해야될 분야가 많이 있다. 아래의 테이블을 확인해보면 최대 50개 만 연구된 것을 알 수 있다.

![History of Face Anti-Spoofing](../History_of_FAS.png)

# Background

## Face Spoofing Attacks

Automatic Face Recognition(AFR)에 대한 공격은 크게 digitial Manipulation과 Physical presentation으로 나뉠수 있다. Digital Manipulation은 디지털 환경에서 복제를 하는 것이다. Physical presentation attack은 물리적인 물질을 이용해서 얼굴을 만들어서 공격을 하는 것이다.

![Face Anti-Spoofing Pipeline and Face Spoofing attacks](../FAS_pipeline_and_Face_spooning_attacks.png)

위의 이미지에서 보인것 처럼, FAS를 AFR에게 적용하는 방식은 크게 두가지로 나뉩니다.

 * 하나는 병렬연산입니다. 이때 FAS와 AFS를 따로 Score를 계산을 합니다. 그 이후에 그 두개의 Score를 결합하여 이미지가 실제 이미지인지 아니지 파악하는 것입니다.
 * 다른 하나는 직렬연산입니다. 처음에 PA을 detect하고 Spoofing 이미지를 Reject합니다. 그 뒤에 Face Recognition을 합니다.

그 아래에 있는 이미지(b)는 여러가지 spoofing attack type을 표현합니다. 이러한 방식은 공격하는 사람의 방식에 따라서 두가지로 나뉩니다.

 * Impersonation : 다른사람의 얼굴을 사용해서 AFR을 속이는 방식입니다. 사진, 디지털 화면, 그리고 3D 마스크등을 사용하는 방식입니다.
 * Obfuscation : 얼굴의 일부분을 가리는 방식으로 공격자의 identity를 속이는 방식입니다. 짙은 화장을 하거나, 선글라스를 끼거나, 아니면 가발을 쓰는 방식입니다.

또한 Geometric property로 PA를 분류해보면, 2D와 3D 공격으로 분류할수 있다.

 * 2D PA는 사진이나 영상을 보여주는 형식으로 되어 있습니다.
   * 평면/곡면 사진
   * 눈,입 구멍이 뚤린 사진
   * 영상
 * 3D PA는 3D 인쇄 기술이 발전하면서 생긴 새로운 공격방식입니다. 이러한 마스크들은 색, 질감 그리고 geometry를 사실적으로 표현합니다. 마스크들을 여러가지 제질로 만들어집니다.
   * Hard/Rigid mask : paper, resin, plaster, plastic
   * flexible soft mask : silicon, latex

얼굴을 가리는 영역에 따라서도 나눌 수 있다.
 * Whole attacks은 가장 흔한 방식의 공격이다.
    * Print photo
    * Video replay
    * 3D mask
 * partial attacks은 흔하지 않은 방식의 공격입니다.
    * Part-cut print photo
    * Eyeclasses
    * Partial tattoo

## Dataset for Face Anti-Spoofing

Deep Learning을 기반으로 하는 방식은 훈련을 할때와 검증할때 다양하고 많은 양의 데이터가 필요합니다.

![Dataset visualiszation](../Visualization_of_Dataset.png)

위의 이미지에서 보는것처럼, 다양한 종류의 데이터들이 있습니다. 위에서 보여진것 처럼 RGB를 비슷한 환경에서 찍은 사진이 있을 수 있고, 아니면 여러가지 센서들을 이용해서 데이터들이 있을 수 있습니다.

![public dataset for Face Anti-Spoofing](../Public_Dataset.png)

이 위의 이미지는 여러가지 데이터셋을 표현하는 이미지입니다. I/V는 각각 이미지와 비디오를 뜻합니다.

이러한 데이터셋을 만드는 세가지 트랜드가 있습니다.

 * Large scale data amount
   * CelebA-Spoof와 HiFiMask dataset은 600000개 이상의 이미지와 50000개 이상의 비디오를 가지고 있습니다. 이 데이터의 대부분은 PA들입니다.
 * 다양한 데이터 분포
   * 일반적인 사진이나 비디오 뿐만아니라 새로운 공격 방식들도 많아졌습니다.
 * mutliple modalities and specialized sensors
   * 일반적인 RGB데이터 뿐만아니라 다양한 센서를 사용합니다.
     * NIR
     * Depth
     * Thermal
     * SWIR
     * Other (Light field Camera)

## Evaluation Metrics

FAS system은 bonafide와 PA의 acceptance와 rejection을 기반으로 연산합니다. 가장 기본적인 두가지는 False Rejection Rate와 False Acceptance Rate를 가장 많이 쓴다.

FAR는 Spoffing attack을 사실이라고 판정한 비율이다. FRR는 실제 데이터를 가짜라고 판별한 비율이다.


FAS는 국제적인 기준 ISO/IEC DIS 30107- 3:2017 standards를 기반으로 다양한 시나리오에서의 퍼포먼스를 측정한다.

Intra- 와 Cross-Testing 에서 가장 많이 사용되는 것은 Half Total Error Rate(HTER), Equal Error Rate(EER), 그리고 Area Under the Curve(AUC)이다.

HTER는 FRR과 FAR의 평균을 사용합니다. EER는 HTER의 특별한 값입니다. 이는 FAR과 FRR가 같은 값을 가지고 있을 때를 나타냅니다. AUC는 bonafide와 spoofing 간의 분리도를 나타냅니다.


Attack Presenataion Classification Error Rate(APCER), Bonafide Presentation Classification Error Rate (BPCER) 그리고 Average Classification Error Rate (ACER)
또한 intra-dataset testing에 사용됩니다.

BPCER와 APCER 는 각각 bonafide classification error와 Attack classification Error를 의미합니다. ACER는 BPCER와 APCER의 평균값고, intra-dataset의 reliability를 평가하는데 사용됩니다.

## Evaluation Protocols

![Four evaluation protocols](../four_evaluation_protocols.png)

### Intra-Dataset Intra-Type Protocol

Intra-dataset intra-type protocol는 환경의 변화가 거의 없는 상황에서 FAS Dataset를 평가하는 방식으로 많은 환경에서 사용되고 있습니다. 

Training과 Testing data가 같은 Dataset에서 추출되었기에, 그들은 녹화된 환경이나, 객체의 행동에 관하여 비슷한 domain distribution을 가지고 있습니다. Deep Learning 기술의 강한 discriminative feature representation ability 덕분에 좋은 퍼포먼스를 보여줍니다. 이는 외부 환경, 카메라 변화, 그리고 공격방식이 크게 변하지 않는 이상 좋은 결과물을 내줍니다.

### Cross-Dataset Intra-Type Protocol

Cross-dataset은 일반화가 잘 되어 있는지를 확인하는 것입니다. 이 방식은 하나 또는 여러개의 dataset에서 훈련을 하고 unseen datasets을 가지고 결과물을 검증하는 방식을 사용합니다.

### Intra-Dataset Cross-Type Protocol

이 방식은 하나의 공격방식을 제외하고 훈련합니다. 그리고 나중에 그 공격방식을 몰랐을때 그를 분별할 수 있는지 실험하는 방식입니다.

### Cross-Dataset Cross-Type Protocol

Cross Dataset Cross Type Protocol은 일반화를 확인하는 방법으로 unseen domain과 unknown attack types에 대하여 검증하는 방식입니다.

# Deep FAS with Commercial RGB Camera

Comercial RGB camera is widely used in many real-world application scenarios. There are three main categories for exisiting deep learning based FAS methods using comercial RGB camera: Hybrid type learning methods combining both handcrafted and deep learning features; common end-to-end supervised deep learning methods; generalized deep learning methods.

![Topology of the deep learning based FAS methods](../Topology_of_DL_FAS.png)

![Chronological overview of the milestone deep learning based FAS methods using commercial RGB camera](../Chronological_overview_of_DL_FAS_RGB.png)

## Hybrid Method

![Table 3](../Table_3.png)

DL and CNN achieved great success in many computer vision tasks. However for FAS, they suffer the overfitting problem due to the limited amount and diversity of the training data. Handcrafted features have been proven to be discriminative to distinguish bonafide from PAs. Some recent works combine handcrafted features with deep features for FAS. These Hybrid methods can be separated into three main categories.

![Hybrid Frameworks for FAS](../Hybrid_Frameworks_for_FAS.png)

The first method is to extract handcrafted features from inputs then employ CNN for semantic feature representation.

The Second method is to extract handcrafted features from deep confolutional features.

The thrid method is to fuse handcrafted and deep convolutional features fro more generic representation.

## Common Deep Learning Method

Common deep learning based methods directly learn the mapping functions from face inputs to spoof detection. Common deep learning frameworks usually include

 * direct supervision with binary cross-entropy loss
 * pixel-wise supervision with auxiliary task
 * generative models.

![Typical end-to-end deep learning frameworks for FAS](../Typical_E2E_DL_FW_FAS.png)

### Direct Supervision with Binary Cross-Entropy loss

FAS can be intuitively treated as a binary classification task. Numerous end-to-end deep learning methods are directly supervised with binary cross-entropy(CE) loss as well as other extented losses.

![Summary of the representative common deep learning based FAS methods with binary cross-entropy supervision](../Summary_of_common_DL_FAS_binary_CE.png)

Researchers have proposed various network architecture supervised by binary CE. There are few works modifying binary CE loss to provide more discriminative supervision signals

## Pixel-wise Supervision

Pixel-wise supervision can provide more fine-graned and contextual task-related clues for better intrinsic feature learning. There are two type of pixel-wise supervision. One based on the physical clues and discriminative design philosophy, auxiliary supervision signals. The other generative models with explicit pixel-wise supervision are recently utilized for generic spoofing parttern estimation.

![Summary of the representative common deep learning based FAS methods with pixel-wise supervision](../Summary_of_common_DL_FAS_PW_supervision.png)

### Pixel-wise supervision with Auxiliary Task

According to human knowledge, most PAs(e.g. plain printed paper and electronic screen) merely have no genuine facial depth information. As a result, recent works adopt pixel-wise pseudo depth labels to guide the deep models, enforcing them predict the genuine depth for live samples, while zero maps for the spoof ones. Another method is to use binary mask.

### Pixel-wise supervision with Generative Model
 
Mine the visual spoof patterns existing in the spoof samples, aming to provide a more intuitive interpretation of the sample spoofness.

## Generalizing Deep Learning Method

Common end-to-end deep learning based FAS methods might generalize poorly on unseen dominant conditions and unknown attack types. Therefore these methodes are unreliable to be applied in practival applications with strong security needs. There are two methods on enhancing generalization capacity of the deep FAS models. One is domain adaptation and generalization techniques. The other is zero/few-shot learning and anomaly detection.

### Generalization to Unseen Domain

![Framework comparison among domain adaptation, domain generalization, and federate learning](../FW_comparison_DA_DG_FL.png)

Domain adaptation technique leverage the knowledge from target domain to bridge the gap between source and target domains. Domain generalization helps learn the generalized feature representation from multiple source domain directly withous any access to target data. Federate learning framework is introduced in learning gneralized FAS models while preserving data privacy.

![Summary of the representative generalized deep learning FAS methods to unseen domain](../Summary_of_generalized_DL_FAS_unseen_domain.png)

#### Domain Adaptation

Domain Adaptation technique alleviate the discrepancy between source and target domains. The distribution of source and target features are matched in a learned feature space. If the features have similar distrivutions, a classifier trained on features for the source samples can be used to classify the target live/spoof samples. However, it is difficult and expensive to collect a lot of unlabed data for traning.

#### Domain Generalization

Domain generalization assumes that there exists a generalized feature space underling the seen multiple source domains and the unseen but related target domain. Learned model from seen source domain can generalize well to the unseen target domain.

This is a new hot spot in recent years. Domain generalization benefits FAS models to perform well in unseen domain, but it is still unknown whether it deteriorates the discrimination capability for spoofing detection under the seen scenarios.

#### Federate Learning

A genearlized FAS model can be obtianed when trained with face images from different distribution and different types of PAs. Federate learning, a distributed and privacy-preserving machine learning techinques, is introduced in FAS to simulataneously take advantage of rich live/spoof information available at different data owners while maintaining data privacy.

To be specific, each trains its own FAS model. Server learns a global FAS model by iteratively aggregating model updates. Then the converged global fAS model would be utilized for inference.

This solve the privacy of data sets, but neglects the privacy issues in the model level.

### Generalization to Unknown Attack Types

FAS models are vulnerable to emerging novel PAs. There are two general way of detecting unknown spoofing attack detection. One is zero/few-shot learning. The other is anomaly detection.

![Summary of the generalized deep learning FAS methods to unknown attack types](../Summary_of_generalized_DL_FAS_unknown_attack_types.png)

#### Zero/Few-Shot Learning

Zero-Shot Learning aims to learn generalized and discriminative features from the predefined PAs for unknown novel PA detection. Few-Shot learning aims to quickly adapt the FAS model to new attacks by learning from both the predefined PAs and the collected very few samples of the new attack.

The performance drops obviously when the data of the target attack types are unavailable for adaptation. The failed detection usually occurs in the challenging attack types, which share similar appearance distribution with the bonafide.

#### Anomaly Detection

Anomaly detection for FAS assumes that the live samples are in a normal class as they share more similar and compact feature representation while features from the spoof samples have large distribution discrepancies in the anomalous sample space due to the high variance of attack types and materials. Anomaly detection trains a reliable one-class classifier to cluster the live samples accurately. Then any samples outside the margin of  the live sample cluster would be detected as the attack.

Anomaly detection based FAS methods would suffer from discrimination degradation.

# Deep FAS with Advanced Sensor

![Comparison with sensor/Hardware for FAS under 2 environment and three attack types](../Comparison_with_sensor_hardware_for_FAS.png)


## Uni-Modal Deep Learning upon Specialized Sensor.

![Summary of the representative deep learning FAS methods with specialized sensor/hardware inputs.](../Summary_of_represntative_DL_FAS_with_specialized_sensor.png)

## Multi-Modal Deep Learning

![Summary of the multi-modal deep learning FAS methods](../Summary_of_multi_modal_DL_FAS.png)

Multi-modal FAS with acceptable costs are increasedly used in real-world application.

### Multi-Modal Fusion

Mainstream multi-modal FAS methods focus on feature level fusion strategy. There are few works that consider input-level and decision level fusions.

### Cross-Modal Translation

The missing modality issues can be raised when using multi-modal FAS. Therefore some uses cross-modal translation techniques to generate missing modal data.