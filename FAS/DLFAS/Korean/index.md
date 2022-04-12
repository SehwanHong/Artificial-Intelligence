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

Attacks on automatic face recognition (AFR) system usuallay divide into two categories: digital manipulation and physical presentation attacks. Digital manipulations uses digital virtual domain. Physical presenatation attack present face upon physical mediums in front of physical face presentation attacks.

![Face Anti-Spoofing Pipeline and Face Spoofing attacks](../FAS_pipeline_and_Face_spooning_attacks.png)

As Shown in the images, there are two ways to integrate FAS with AFR:

 * Parallel fusion with predicted scores from FAS and AFR. The combined new final score is used to determine if the sample comes from a genuine user or not
 * Serial scheme for early face PAs detection and spoofing rejection, thus avoiding face accessing the subsequent face recognition phase.

The image (b) represent some spoofing attack types. According to attacker's intention, face PAs can be divided into two typical cases

 * Impersonation : spoof to be recognized as someone else via copying genuine user's factial attributes to special mediums such as photo, electronic screen, and 3D mask
 * Obfuscation : hide or remove the attacker's own identity using various methods such as glasses, makeup, wig, and disguised face

Based on the Geometric property, PAs are broadly classified into 2D and 3D attacks.

 * 2D PAs are carried out by presenting facial attributes using photo or video to the sensors
    * Flat/Wrapped printed photos
    * Eye/mouth-cut photos,
    * Digital replay of videos
 * 3D PAs have become a new type of PAs as 3D printing technology. Face Masks are more realistic in terms of color, texture, and geometry. These masks are made of different materials
    * Hard/Rigid mask : paper, resin, plaster, plastic
    * flexible soft mask : silicon, latex

Based on the facial region covering, PAs be also separated as whole or partial attacks.

 * Whole attacks are common
    * Print photo
    * Video replay
    * 3D mask
 * partial attacks are uncommon
    * Part-cut print photo
    * Eyeclasses
    * Partial tattoo

## Dataset for Face Anti-Spoofing

Large-scale and diverse dataset are pivotal for deep learning based methods during both training and evaluating phase.

![Dataset visualiszation](../Visualization_of_Dataset.png)

As shown in the images thereare different kinds of dataset, It might contain just RGB images with similar setting or might have multiple modalities.

![public dataset for Face Anti-Spoofing](../Public_Dataset.png)

I and V in the "#Live/Spoof" denotes 'images' and 'video'

There are three main trends for dataset progress:

 * Large scale data amount
    * CelebA-Spoof and HiFiMask dataset contains more than 600000 unages and 50000 videos, where most of them are PAs
 * Diverse data distribution
    * Besides common print and replay attacks (recorded in indoor), novel attack types are introduced.
 * mutliple modalities and specialized sensors
    * Apart from traditional RGB, recent dataset consider other sensors
        * NIR
        * Depth
        * Thermal
        * SWIR
        * Other (Light field Camera)

## Evaluation Metrics

FAS system focuse on the concept of bonafide and PA acceptance and rejection. Two basic metrics False Rejection Rate and False Acceptance Rate are widely used.

FAR is the ratio of incorrectly accepted spoffing attacks. FRR is the ratio of increectly rejected live accesses.

FAS follows ISO/IEC DIS 30107- 3:2017 standards to evaluate the performance of the FAS systems under different senarios.

The most commonly used metrics in intra- and cross-testing scenarios ais Half Total Error Rate(HTER), Equal Error Rate(EER), and Area Under the Curve (AUC).

HTER is found out by calculating the average of FRR and FAR. EER is a specific value of HTER at which FAR and FRR have equal values. AUC represents the degree of separablility between bonafide and spoofings.

Attack Presenataion Classification Error Rate(APCER), Bonafide Presentation Classification Error Rate (BPCER) and Average Classification Error Rate (ACER) suggested in ISO standard are also used for intra-dataset testing.

BPCER and APCER measure bonafide aand attack classification error rates, respectively. ACER is calulated as the mean of BPCER and APCER, evaluating the reliability of intra-dataset performance.

## Evaluation Protocols

![Four evaluation protocols](../four_evaluation_protocols.png)

### Intra-Dataset Intra-Type Protocol

Intra-dataset intra-type protocol has been widely used in most FAS datasets to evaluate the model's discrimination abuility for spoofing detection under scenarios with slight domain shift.

As the training and testing data are from the same datasets, they share similar domain distribution in terms of the recording environment, subject behavior. Due to strong discriminative feature representation ability via deep learning, many methods have reached satisfied performance (<5% ACER) under small domain shifts about external encironments, attack mediums and recording camera variation.

### Cross-Dataset Intra-Type Protocol

Cross-dataset level domain generalization ability measurement. This protocol tests on one or several datasets and then tests on unseen datasets.

### Intra-Dataset Cross-Type Protocol

The protocol adopts 'leave one attack type out' to validate the model's generalization for unknown attack types. One kind of attack type only appears in the testing stage.

### Cross-Dataset Cross-Type Protocol

Cross Dataset Cross Type Protocol to measure the FAS model's generalization on both unseen domain and unknown attack types.

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