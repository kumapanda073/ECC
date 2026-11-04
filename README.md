# ECC用のシミュレーションコード: SNS テキストを「人センサ」として用いた群集状態推定

## 人由来の観測情報を用いた人流推定

人の発する「ここめっちゃ混んでる」「全然人がいない」といったテキストを、物理センサと同列に扱える観測値として状態推定に取り込むという研究のコードです。カルマンフィルタにより物理センサと融合し、通路の区間ごとの群集密度・歩行速度を推定します。

テキストはBERTベースの2ヘッドネットワークで定量化します。

分類ヘッド: テキストを M 段階のカテゴリに分類し、そのカテゴリ代表値を観測値 y として出力。同時に、分類の確信度の低さ（解釈の曖昧さ）を分散 σ²_cls として算出
分散ヘッド: 分類の確信度には現れない残差成分（量子化誤差など）σ²_var を推定

観測ごとに R = σ²_cls + σ²_var をカルマンフィルタの観測ノイズ分散として動的に設定するのが核心です。曖昧なテキストや状態と無関係なテキストほど R が大きくなり、カルマンゲインを通じて推定への寄与が自動的に抑えられます。

- **観測分散 R** = 分類確率から計算した分散 (`dist_var`) + 分散ヘッドの出力。
  無関係なテキストには大きな分散を出すよう学習させている。
- 比較として、ガウス NLL で学習した **回帰モデル** (値と分散を直接出力) も実装している (`vsAI/`)。

## ディレクトリ構成

```
ECC/
├── README.md
├── requirements.txt
├── common/                          モデル・データ処理の共通部品
│   ├── models/encoder.py                BERT エンコーダ ([CLS] 埋め込みを出力)
│   └── utils/
│       ├── dataset.py                   ラベル (5 クラス) 変換・Dataset / DataLoader
│       └── preprocess.py                テキストの前処理
│
├── data/                            学習用テキストデータ
│   ├── train_den.json / val_den.json    密度ラベル付きテキスト
│   ├── train_vel.json / val_vel.json    速度ラベル付きテキスト
│   └── split_data.py                    train / val への分割スクリプト
│
├── training/                        テキスト定量化モデルの学習
│   ├── dual_head/                       提案手法: 分類ヘッド + 分散ヘッド
│   │   ├── models/                          モデル定義
│   │   ├── train.py / train_velocity.py     学習 (密度 / 速度)
│   │   ├── evaluate.py                      検証データでの評価
│   │   └── predict.py                       1 文を入力して推定結果を確認
│   └── gaussian_nll/                    比較手法: ガウス NLL による回帰モデル
│       ├── regression_model.py              モデル定義
│       └── train_density.py / train_velocity.py
│
├── simulation/                      提案手法のシミュレーション評価
│   ├── main.py                          実験の入口
│   ├── config.py                        区間数・ノイズ・センサ配置などの設定
│   ├── model.py                         状態空間モデル (A, B, Q)
│   ├── simulation.py                    真値生成とカルマンフィルタの実行
│   ├── kalman.py                        カルマンフィルタ 1 ステップ
│   ├── sensors.py                       物理センサ / 人センサの観測生成
│   ├── nlp_bridge.py                    テキスト → 観測値・観測分散 R
│   ├── text_pool.json                   シミュレーションで投稿されるテキスト
│   ├── noisy_pool.json                  状態と無関係なテキスト
│   ├── csv/                             状態遷移行列 A
│   ├── experiments/                     各実験 (誤り率 / 無関係テキスト率 / 時系列)
│   ├── plotting/                        図の描画
│   └── figure/                          出力した図と集計 CSV
│
└── vsAI/simulation_v2/              分類 + KF (提案) と 回帰 / 回帰 + KF の比較実験
    ├── main.py                          実験の入口
    ├── regression_bridge.py             テキスト → 回帰モデルの出力
    ├── estimators.py                    3 手法の推定
    ├── sensing.py                       人センサの観測生成
    ├── calibration.py                   回帰の観測分散の調整
    ├── error_metrics.py                 RMSE / MAE
    ├── plots.py                         図の描画
    ├── paths.py                         パス設定
    └── results/                         出力した図
```

## 実行方法

### 1. 環境構築

```bash
git clone https://github.com/kumapanda073/ECC.git
cd ECC
pip install -r requirements.txt
```

<!-- TODO(本人): 使用した Python のバージョンを記入 (例: Python 3.x) -->

初回実行時に日本語 BERT (`cl-tohoku/bert-base-japanese-v3`) が Hugging Face から自動でダウンロードされます。

### 2. モデルの学習

学習済みモデル (`.pt`, 1 個 約 430MB) はサイズの都合でリポジトリに含めていないため、最初に学習します。
各スクリプトは 10 エポック学習し、エポックごとのモデルをスクリプトと同じフォルダ内に保存します。

```bash
# 提案手法 (分類ヘッド + 分散ヘッド)
python training/dual_head/train.py               # 密度 → training/dual_head/checkpoints/
python training/dual_head/train_velocity.py      # 速度 → training/dual_head/checkpoints_velocity/

# 比較手法 (回帰モデル)。vsAI の比較実験で使用
python training/gaussian_nll/train_density.py    # → training/gaussian_nll/checkpoints_regression_density/
python training/gaussian_nll/train_velocity.py   # → training/gaussian_nll/checkpoints_regression_velocity/
```

学習したモデルの確認:

```bash
python training/dual_head/evaluate.py    # 検証データでの正解率などを評価
python training/dual_head/predict.py     # テキストを入力して推定密度と分散を表示 (q で終了)
```

### 3. シミュレーション (提案手法の評価)

```bash
python simulation/main.py               # すべての実験
python simulation/main.py error_rate    # テキストの選択ミス率を変える実験
python simulation/main.py noise_rate    # 無関係テキストの割合を変える実験
python simulation/main.py timeseries    # 時系列の推定デモ
python simulation/main.py replot        # 保存済みデータから図だけ再描画
```

結果は `simulation/figure/` に保存されます。
区間数・試行回数・センサ配置などの条件は `simulation/config.py` で変更できます。

### 4. 分類 vs 回帰の比較実験

```bash
python vsAI/simulation_v2/main.py
```

結果は `vsAI/simulation_v2/results/` に保存されます。

## 論文

現在査読中です。

## 作者

<!-- TODO(本人): 名前・所属 (載せたい範囲で) -->
