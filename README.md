# Exploring Highly Quantised Neural Networks for Intrusion Detection in Automotive CAN

## Overview
This repository implements a custom-quantised multi-layer perceptron (CQMLP) neural network for real-time intrusion detection on automotive Controller Area Networks (CAN). The project focuses on using a single, low-power multi-class classification model capable of detecting multiple attack types simultaneously, reducing the need for multiple models and optimizing resource usage.

### Key Features
- **Multi-Class Detection**: Detects multiple attack vectors (DoS, fuzzing, spoofing) using a single neural network.
- **Custom Quantisation**: Employs 2-bit precision for weights and activations, significantly reducing resource and power consumption.
- **FPGA Deployment**: Utilizes AMD/Xilinx Brevitas for training and FINN for hardware acceleration targeting the XCZU7EV device.
- **High Accuracy**: Achieves 99.9% detection accuracy on public CAN attack datasets.
- **Low Latency and Energy Efficiency**: Operates at a latency of 0.11 ms per message and consumes only 0.23 mJ per inference.
---

## Prerequisites

### Hardware
- XCZU7EV or compatible FPGA.

### Software
- AMD/Xilinx Vitis-AI, Brevitas, and FINN toolflows.
- Python 3.7+.

### Libraries
- PyTorch with Brevitas support.
- TensorFlow 2.x.
- NumPy.

---

## Results
- **Detection Accuracy**: 99.9% on DoS, fuzzing, and spoofing attacks.
- **Latency**: 0.11 ms per message window.
- **Energy Consumption**: 0.23 mJ per inference.

---

## Citation
If you use this work in your research, please cite:

```
@inproceedings{khandelwal2023exploring,
  title={Exploring Highly Quantised Neural Networks for Intrusion Detection in Automotive CAN},
  author={Khandelwal, Shashwat and Shreejith, Shanker},
  booktitle={2023 33rd International Conference on Field-Programmable Logic and Applications (FPL)},
  pages={235--241},
  year={2023},
  organization={IEEE}
}

```
