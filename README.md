# DiODeFusion - Diversity-Informed Fusion for *Sclerotinia sclerotiorum* and White Mold Detection
DiODeFusion (Diversity-Informed Object Detection Fusion) is a framework for improving object detection. It selects models based on diversity measures and fuses the predictions using voting strategies.

<!--
# Detection and classification of whiteflies and development stages on soybean leaves images using an improved deep learning strategy
This repository has the aim storing all artefacts (image dataset) used in the research related to the paper.
-->

## Authors

- Rubens de Castro Pereira <sup>a,b,c,e</sup>
- Díbio Leandro Borges <sup>d</sup>
- Murillo Lobo Jr. <sup>b</sup>
- Ricardo da Silva Torres <sup>e</sup>
- Hélio Pedrini <sup>a</sup>

<sup>a</sup> Institute of Computing, University of Campinas, Campinas, 13083-852, SP, Brazil

<sup>b</sup> EMBRAPA Rice and Beans, GO-462, km 12, Santo Antônio de Goiás, 75375-000, GO, Brazil

<sup>c</sup> Institute of Informatics, Federal University of Goiás, Goiânia, 74690-900, GO, Brazil

<sup>d</sup> University of Brasília, Department of Computer Science, Brasília, 70910-900, DF, Brazil

<sup>e</sup> Artificial Intelligence Group, Wageningen University and Research, Droevendaalsesteeg 1, Wageningen, 6708PB, Netherlands

## Abstract

The white mold caused by the soilborne pathogen *Sclerotinia sclerotiorum* affects hundreds of plant hosts around the world. Its early detection is important to leverage effective and timely responses that mitigate crop yield losses. This paper addresses the automatic detection of white mold based on computer vision methods. We introduce a Diversity-driven Object Detection Fusion framework, DiODeFusion, that leverages the complementary views provided by multiple detectors to improve prediction results. DiODeFusion relies on bounding-box-based diversity measures to determine the most promising detectors for use in fusion. Experiments were conducted on a recently created SWM dataset, considering the assessment of multiple diversity measures, detector selection approaches, and bounding-box fusion strategies. Experimental results show that DiODeFusion achieves gains of up to 2.0\% in terms of the F1 metric using only three detectors, compared to strong fusion baselines that employ all available detectors.

## Artifacts

- Weight Files

- Source Code

<!-- 
Images dataset used in training [:bug:](https://github.com/rubenscp/Whiteflies_Dataset/tree/main/dataset_for_training)
Images dataset used in whiteflies detection [:bug:](https://github.com/rubenscp/Whiteflies_Dataset/tree/main/dataset_for_detection_test)
Images used in the paper [:arrow_right:](https://github.com/rubenscp/Whiteflies_Dataset/tree/main/images_of_the_paper)
--> 