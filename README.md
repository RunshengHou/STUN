<div align=center>

# 🚦 STUN: RL-based Optimization of Linux Scheduler Params

</div>

> 📘 This repository provides a **non-official Python implementation** of the paper titled:  
**"STUN: Reinforcement-Learning-Based Optimization of Kernel Scheduler Parameters for Static Workload Performance"**.  

> The original paper is published in *Applied Sciences* and can be accessed [here](https://mdpi-res.com/d_attachment/applsci/applsci-12-07072/article_deploy/applsci-12-07072.pdf?version=1657715467).

## 🌍 Environment

- 🐧 Ubuntu 18.04 with Linux kernel 4.15 
- 🐍 Python 3.10

## 📂 Dataset

Follow the steps below to download and prepare the BioID Face Database for use:

1. **Create a directory for the dataset**

    ```bash
    mkdir dataset && cd dataset
    ```

2. **Download the BioID Face Database**

    ```bash
    wget https://www.bioid.com/uploads/BioID-FaceDatabase-V1.2.zip
    ```

3. **Extract the dataset**

    ```bash
    unzip BioID-FaceDatabase-V1.2.zip -d BioID-FaceDatabase
    ```

4. **Clean up unnecessary files**

    ```bash
    rm BioID-FaceDatabase-V1.2.zip BioID-FaceDatabase/*.eye BioID-FaceDatabase/*.txt
    ```

## 🚀 Getting Started

1. **Clone the repository**

    ```bash
    git clone https://github.com/runshenghou/stun
    cd STUN
    ```

2. **Install dependencies**

    ```bash
    pip install numpy gym tqdm opencv-python
    ```

3. **Run the scripts**

    ```bash
    python ./stun.py
    ```

## 📄 License

This project is licensed under the terms of the [**GPL 3.0 License**](/LICENSE).