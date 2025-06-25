Here is the English translation:

This is my graduation project, titled "An Intelligent Diagnosis System for Biventricular Cardiomyopathy Based on Multimodal Data Fusion."

In recent years, with the development of machine learning, deep learning, and medical imaging technologies, significant progress has been made in the early detection and risk prediction of heart diseases. Taking the high-incidence biventricular h
eart as an example, this paper proposes an intelligent diagnosis system for biventricular cardiomyopathy based on multimodal data fusion, aiming to improve diagnostic efficiency and patient prognosis.

First, a blood oxygen and heart rate collector based on STM32F103ZET6 and MAX30102 is designed. By optimizing for low latency, low power consumption, and interrupt response, the system achieves efficient and reliable data acquisition, processing, and uploading, with real-time physiological parameter display via an OLED screen. The device is compact, non-invasive, easy to use, and portable, making it suitable for daily health monitoring.

Secondly, an improved neural network architecture for semantic segmentation is studied for ventricular imaging segmentation. This paper adopts a network architecture that combines the advantages of DeepLabV3 and UNet, enabling effective extraction of key cardiac structural information. Experimental results show that this architecture is not only efficient and improves the accuracy of medical segmentation, but also uses the segmentation results as evaluation indicators for cardiac morphology.

Next, the time-varying characteristics of the heart are analyzed from the perspective of cardiac electro-mechanical-fluid coupling simulation. To this end, a simplified cardiac geometry is constructed and multiphysics simulation is performed using COMSOL MultiPhysics software. Experimental results indicate that this coupling model can effectively simulate the mechanical response and electrophysiological behavior of cardiac tissue under electrical stimulation.

Finally, the above components are integrated into a complete system, which includes a front-end interface (Tkinter), database (SQLite), and image processing (OpenCV, TransDeeplib-UNet), as well as deep learning-based medical image segmentation and an XGBoost diagnostic model. In addition, the system features remote consultation capabilities and automatic report generation. The modular design of the entire system allows each function to be developed and upgraded independently, while also working together to form a complete diagnostic workflow.

In summary, this study integrates machine learning, feature engineering, and advanced Transformer and UNet networks to enhance medical image segmentation performance, employs a high-sensitivity microcontroller system for data acquisition, processing, and uploading, and finally achieves automated drug delivery through a knowledge graph approach. Thus, an intelligent diagnosis system for biventricular cardiomyopathy based on multimodal data fusion is realized. This research not only promotes the development of personalized medicine but also provides new ideas for interdisciplinary collaboration, contributing to the development of more intelligent and personalized health management systems.

Keywords: Cardiomyopathy; Data Fusion; Deep Learning; Electro-mechanical-fluid Coupling; System Integration
