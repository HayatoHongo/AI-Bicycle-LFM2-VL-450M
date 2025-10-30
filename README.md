# 🇯🇵 LFM2-VL Japanese Fine-tuning (Docker + A100 + CUDA 12.6)

日本語VQA（視覚言語）データセットを用いて  
`LiquidAI/LFM2-VL-450M` モデルを微調整（Fine-tuning）する環境です。  
Dockerベースで、GPU環境（A100など）に最適化されています。

---

## 🧱 環境構成

| コンポーネント | バージョン / 備考 |
|----------------|------------------|
| CUDA | 12.6 |
| Python | 3.11 |
| PyTorch | 2.8.0+cu126 |
| Transformers | 4.55.0 |
| TRL | 0.22.2 |
| PEFT | 0.17.1 |
| Accelerate | 1.10.1 |
| Datasets | 4.0.0 |

---



```bash
git clone https://github.com/HayatoHongo/LFM2-VL-450M-JA-Instruct.git
```

インストール方法（Ubuntuの場合）

```
sudo apt update
sudo apt install git-lfs
```

インストール後、初期化します：
```
git lfs install
```

その後、LFSファイルを取得します：
```
git lfs pull
```


## 🚀 ビルド & 実行

### 1️⃣ Docker イメージ pull or ビルド


#### pull 
For ubuntu/x86 arch
```bash
sudo docker pull hayatohongo/lfm2-vl-ja:cu126-20251008-amd64 
```

```bash
sudo docker run --gpus all -it --rm \
  --ipc=host \
  -v /home/ubuntu/REPO:/workspace \
  -v /home/ubuntu/IMAGE_PATH_FILESYSTEM:/workspace/images \
  --name lfm2-vl-ja-train \
  hayatohongo/lfm2-vl-ja:cu126-20251008-amd64
```

上記のDockerイメージではwandbは無効化されているので、必要に応じてインストールしてください。

```bash
pip install wandb
unset WANDB_DISABLED
```

#### build (上記が動かなければ)

```bash
sudo docker build -t lfm2-vl-ja:cu126 .
````

---

### 2️⃣ コンテナを起動（GPU使用・学習用）

```bash
sudo docker run --gpus all -it --rm \
  --ipc=host \
  -v /home/ubuntu/LFM2-VL-450M-JA-Instruct:/workspace \
  -v /home/ubuntu/llava-virginia/images:/workspace/images \
  --name lfm2-vl-ja-train \
  hayatohongo/lfm2-vl-ja:cu126-20251008-amd64
```

## hugging face への訓練済みモデルのpush 

### コンテナ内で実行

```bash
pip install huggingface_hub
```

---

## ✅ 推奨動作環境

| 項目     | 推奨                               |
| ------ | -------------------------------- |
| GPU    | NVIDIA A100 80GB × 8             |
| CUDA   | 12.6                             |
| Driver | >= 550.54                        |
| Docker | >= 24.0                          |
| OS     | Ubuntu 22.04 / Amazon Linux 2023 |

---

## 🧾 開発メモ

* `requirements.txt` と `Dockerfile` により完全再現可能。
* Hugging Face キャッシュは `/workspace/.cache` に保存。

---

## 🧑‍💻 作者

**Hongoh Hayato**
Fine-tuned LFM2-VL 日本語モデル開発者