# 🇯🇵 AI Bicycle LFM2-VL Japanese Fine-tuning (Docker + A100 + CUDA 12.6)

[Model Card](https://huggingface.co/HayatoHongo/lfm2-vl-ja-finetuned-enmt1ep-jamt10eponall-vqa)

## Demo is available on Colab! 
[Demo notebook](https://colab.research.google.com/drive/193EhKhY_zUtKiBwoZbXwL9vjWAECBcV4?usp=sharing)

## デモ動画

ディズニー映画『カーズ』に着想を得て、自転車とおしゃべりできるようなマルチモーダルAIを開発しました。<br>
450Mパラメータの小規模モデルなので、スマートフォンに組み込めばローカルでの高速推論が可能です。<br>
以下は Streamlit の クラウドサーバーでデプロイした際のデモ動画です。<br>
動画は4倍速にしています。ローカルでの推論に切り替えれば、さらなる高速応答が期待されます。<br>

https://youtu.be/XmSLuOB5WK4


モデルは text + image -> text をサポートしています。<br>
音声文字起こし、およびテキスト読み上げと組み合わせることで、<br>
audio -> text + image -> text -> audio により、運転中での会話も可能となります。<br>

また、例えば5分おきに画像を撮影してモデルに送信することで、ユーザーからのプロンプトがない場合でも、<br>
自転車からユーザーに話しかけることができます。<br>


## 🧑‍💻 Author

Hayato Hongo
Developer of the fine-tuned LFM2-VL Japanese model

Special thanks to **Leo Paul** for deploying the model to apps and creating demo videos,
and to **Rikka Botan** for continuous and stable contributions to the dataset and deck slides.


## ベースモデル

日本語データセットを用いて  
`LiquidAI/LFM2-VL-450M` モデルを微調整（Fine-tuning）して開発しました。
Dockerベースで、GPU環境（A100）に最適化されています。


## 🧭 トレーニングは3段階のパイプラインを経ています

本プロジェクトでは、3つのステージ（Stage 1〜3）を順に実行することでモデルを構築します。
各ステージの詳細や構成図については、以下のリンクから確認できます。

🔗 **デモ／構成図（Canva）**
[https://www.canva.com/design/DAG1jHz8MAM/aWNfQN6LETEhQxzBroJswQ/edit?utm_content=DAG1jHz8MAM&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton](https://www.canva.com/design/DAG1jHz8MAM/aWNfQN6LETEhQxzBroJswQ/edit?utm_content=DAG1jHz8MAM&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)


##　⚠️ 注意

以下に再現実験の手順を示します。
ただし、ユーザーテストをしていないため、間違っている部分もあるかもしれません。
致命的な間違いはないと思いますが、例えばパスやファイル名の指定などに軽微な誤りがある可能性があります。
その場合はご自身で対応してください。ご不便をおかけしますがお願いいたします。

---

## startar guide

```bash
git clone https://github.com/HayatoHongo/AI-Bicycle-LFM2-VL-450M
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

---

## 📦 データセットのダウンロード

以下のリンクから統合データセットを取得してください：

🔗 [HayatoHongo/AI-Bicycle-LFM2-VL-450M](https://huggingface.co/datasets/HayatoHongo/AI-Bicycle-LFM2-VL-450M)

---

### 📁 データ配置

| ステージ        | ディレクトリ                          | 含まれるファイル                                                                                              |
| ----------- | ------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Stage 1** | `stage1_en_multiturn/data`      | `en_multiturn.jsonl`                                                                                  |
| **Stage 2** | `stage2_ja_multiturn/data`      | `Japanese_multiturn_soda_kaken_train_131152.jsonl`<br>`Japanese_multiturn_soda_kaken_test_2677.jsonl` |
| **Stage 3** | `stage3_Japanese_VQA_CC3M/data` | `Japanese_VQA_CC3M_train_188789.jsonl`<br>`Japanese_VQA_CC3M_test_3853.jsonl`                         |

---

## 🖼️ Stage 3 用の画像データ

Stage 3（VQAモデル）では、画像データを追加でダウンロードする必要があります。
以下のデータセットから取得してください：

🔗 [HayatoHongo/LLaVA-CC3M-Pretrain-521K](https://huggingface.co/datasets/HayatoHongo/LLaVA-CC3M-Pretrain-521K)

ダウンロードするファイル：
`images.tar.zst`

---

### ⚠️ 注意事項

* ファイルサイズは **約19GB** あり、解凍には **約5分** かかります。
* プロジェクトフォルダ直下に配置しても問題ありませんが、
  大容量のため **ファイルシステム上への保存** を推奨します。
* 解凍時は**マルチスレッド展開**を推奨します。
* 解凍スクリプトは紛失してしまいましたが、ChatGPTなどに「`.zst`ファイルをマルチスレッドで解凍する方法」を尋ねれば対応可能です。

⏱️ **補足**：マルチスレッドを使わない場合、解凍時間が大幅に長くなることがあります。

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


### 1️⃣ Docker イメージ pull or ビルド


#### pull 
For ubuntu/x86 arch
```bash
sudo docker pull hayatohongo/lfm2-vl-ja:cu126-20251008-amd64 
```

```bash
sudo docker run --gpus all -it --rm \
  --ipc=host \
  -v /home/ubuntu/YOUR_REPO:/workspace \
  -v /home/ubuntu/YOUR_IMAGE_PATH:/workspace/images \
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
  -v /home/ubuntu/AI-Bicycle-LFM2-VL-450M:/workspace \
  -v /home/ubuntu/YOUR_IMAGE_PATH:/workspace/images \
  --name lfm2-vl-ja-train \
  hayatohongo/lfm2-vl-ja:cu126-20251008-amd64
```

もちろんです。以下は、READMEにそのまま使えるよう整えた **最初の章（トレーニング手順の概要部分）** です👇

---

## 🚀 トレーニング手順

このリポジトリには、各ステージ（Stage 1〜3）のトレーニングを実行するための起動スクリプトが含まれています。
基本的には、**各ステージの起動コマンドファイルを微調整して、実行するだけでトレーニングが開始**されます。

GPU構成は環境に合わせて調整可能です。
お使いのGPUの数に応じて、スクリプト内のパラメータ（例：`--nproc_per_node` など）を変更してください。

なお、次のステージのために、各ステージについて訓練済みモデルをご自身のHuggingFaceにアップロードしてください。

## hugging face への訓練済みモデルのpush 

### コンテナ内で実行

```bash
pip install huggingface_hub
```

### ログイン / HF_TOKEN を入力

```bash
huggingface-cli login
```

### レポジトリ作成

```bash
huggingface-cli repo create lfm2-vl-ja-finetuned-2025xxxx --type model
```

### checkpointとtraining_args.bin は除いてpush

```bash
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="/workspace/output",
    repo_id="HayatoHongo/lfm2-vl-ja-finetuned-20251008",
    ignore_patterns=["checkpoint-*", "training_args.bin"]
)
PY
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

