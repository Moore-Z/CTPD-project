

```
### Task Instruction:
You are given a patient’s vital signals over a 24 hours window, along with their clinical notes. 
Your job is to predict whether this patient will die within the next 48 hours (in-hospital mortality).

### Dataset Context:
- The dataset is derived from MIMIC-III.
- Vital signals include heart rate, blood pressure, temperature, etc.
- Clinical notes are written by doctors and nurses at different times.
- Missing values in vital signals are imputed or encoded.

### Input Format:
1) A series of reprogrammed embeddings representing time segments of the patient's vitals.
2) A sequence of clinical notes (text) that correspond to certain time points.

### Input time segments series statistics:
(min: {min_val}, max: {max_val}, median: {median_val}, trend: {trend_desc}, top-lags: {lag_values})

### Important Details:
- Each vital-signal embedding has an approximate time range (e.g., T=0–1h, T=1–2h).
- Each clinical note is associated with a time stamp (e.g., T=2h or T=5h).
- The total time window is 0–48 hours from ICU admission.

### [BEGIN: Time-Series Embeddings]
[time=0~1h] <vital_embedding_token1> 
[time=1~2h] <vital_embedding_token2>
[time=2~3h] <vital_embedding_token3>
... 
### [END: Time-Series Embeddings]

### [BEGIN: Clinical Notes]
[Note at T=2h]: "Patient complains of chest pain ..."
[Note at T=6h]: "Doctor updated antibiotic plan ..."
...
### [END: Clinical Notes]
```



# Code Doc

## MIMIC3

### PATHs

`/Users/haochengyang/Desktop/Research/CTPD/MMMSPG-014C/EHR_dataset/mimiciii`:所有刚解压的raw csv文件



root 路径 EHR_dataset/mimiciii_benchmark/root

mimiciii路径 EHR_dataset/mimiciii

### PreProcess

#### Prepare time series features

a. Run the command `cd src/cmehr/preprocess`.

b. The following command takes MIMIC-III CSVs, generates one directory per SUBJECT_ID and writes ICU stay information to data/{SUBJECT_ID}/stays.csv, diagnoses to data/{SUBJECT_ID}/diagnoses.csv, and events to data/{SUBJECT_ID}/events.csv. This step might take around an hour. Here `data/root/` denotes the folder to store the processed benchmark data.

```
python -m mimic3benchmark.scripts.extract_subjects {PATH TO MIMIC-III CSVs} data/root/
```

c. The following command attempts to fix some issues (ICU stay ID is missing) and removes the events that have missing information. About 80% of events remain after removing all suspicious rows.

```
python -m mimic3benchmark.scripts.validate_events data/root/
```

d. The next command breaks up per-subject data into separate episodes (pertaining to ICU stays). Time series of events are stored in {SUBJECT_ID}/episode{#}_timeseries.csv (where # counts distinct episodes) while episode-level information (patient age, gender, ethnicity, height, weight) and outcomes (mortality, length of stay, diagnoses) are stores in {SUBJECT_ID}/episode{#}.csv. This script requires two files, one that maps event ITEMIDs to clinical variables and another that defines valid ranges for clinical variables (for detecting outliers, etc.). Outlier detection is disabled in the current version.

```
python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
```

e. The next command splits the whole dataset into training and testing sets. Note that the train/test split is the same of all tasks.

```
python -m mimic3benchmark.scripts.split_train_and_test data/root/
```

f. The following commands will generate task-specific datasets, which can later be used in models. These commands are independent, if you are going to work only on one benchmark task, you can run only the corresponding command.

选择要训练的任务运行

截取的X的period length在这里设定（24/48h）

```
python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/
python -m mimic3benchmark.scripts.create_decompensation data/root/ data/decompensation/
python -m mimic3benchmark.scripts.create_length_of_stay data/root/ data/length-of-stay/
python -m mimic3benchmark.scripts.create_24h_phenotyping data/root/ data/phenotyping/
python -m mimic3benchmark.scripts.create_multitask data/root/ data/multitask/
```

g. Use the following command to extract validation set from the training set. This step is required for running the baseline models. Likewise, the train/test split, the train/validation split is the same for all tasks.

```
  python -m mimic3models.split_train_val {dataset-directory}
```



- all prepared TS data are stores under `src/cmehr/preprocess/mimic3/data`

- 全处理完后转移到`EHR_dataset/mimiciii_benchmark`内并重命名为缩写eg`ihm`

  

> `listfile.csv`与`Reader`
>
> **`listfile.csv` 是 MIMIC-III Benchmark 中的重要索引文件**，用于：
>
> - **标记哪些 ICU 住院记录**（`episodes`）被选为**训练**、**验证**或**测试样本**
>
> - **`listfile.csv` 作为数据索引文件**，告诉 `Reader`：
>
>   - **用哪个 ICU 数据文件** (`*_timeseries.csv`)，
>   - **截取多长时间窗口**，
>   - **这个样本的标签是什么**。
>
>   **`Reader` 通过 `listfile.csv` 加载 ICU 数据**，并返回数据字典
>
>   ```
>   {
>     "X": ICU数据矩阵,
>     "y": 标签,
>     "t": 时间窗口,
>     "name": 样本名,
>     "header": 特征名
>   }
>   ```
>
> - ```
>   listfile.csv
>   │
>   ├── ICU 住院索引  ───────────────────────►  Reader._load_listfile()
>   │     │                                       │
>   │     ├─ stay = 19149_episode1_timeseries.csv │
>   │     ├─ period_length = 48                   │
>   │     └─ label = 0                            │
>   │                                             ▼
>   │                                      读取 ICU 数据路径
>   │                                             ▼
>   └──────────────────────────────────► Reader.read_example()
>                                          │
>                                          ▼
>                             读取 19149_episode1_timeseries.csv
>   ```







#### Prepare Notes

##### extract_notes.py

- 输入文件的是root folder里的数据不是subtask的
- 对train/test分别进行
- 输出也在root folder里

##### extract_T0.py

- 对train/test分别进行



#### create_iiregular_ts.py

Create pickle files to store multimodal data. At the end, you will have a folder `output_mimic3`, which contains the stored pickle files for each task.

```
python -m mimic3models.create_iiregular_ts --task {TASK}
```

多模态数据预处理脚本，用于：

- **将 MIMIC-III 的 ICU 时间序列转换为不规则时间序列**，
- **离散化时间序列数据**，
- **标准化数据**
- **将文本数据对齐与时间序列对齐**，
- 最终输出**多模态数据集**，支持多种下游任务，如：
  - **住院死亡率预测 (IHM)**、
  - **表型分类 (PHENO)**、
  - **30天再入院预测 (READM)**。

##### **输入**

- 一个针对特定任务的MIMIC-III Benchmark 时间序列数据数据集：

  - **Prepare time series features** 预处理后的数据集，已分为train，val，test。

  （如`train_listfile.csv` `val_listfile.csv` `test_listfile.csv`，`events.csv`、`stays.csv`等）。

  - 路径：

    ```
    MIMIC3_BENCHMARK_PATH/
    ├── ihm/                    # 住院死亡率预测数据
    │   ├── train/              # 训练集数据
    │   ├── val/                # 验证集数据
    │   └── test/               # 测试集数据
    ├── pheno/                  # 表型分类数据
    ├── readm/                  # 再入院预测数据
    └── ...
    ```

    

- 临床笔记数据：

  - **Prepare Notes** 预处理后的 JSON 文件（如`19_1.json`，包含按时间戳排序的临床笔记）。

  - 路径：

    ```
    MIMIC3_BENCHMARK_PATH/
    ├── root                   
    │   ├── train_text_fixed
    │   ├── test_text_fixed
    ```
    
    

- 配置文件

  - **`discretizer_config.json`**：定义了时间序列的通道（如 `Heart Rate`、`Blood Pressure`）及缺失值填充值。
  - **`channel_info.json`**：定义了每个通道的类别及可能取值。

##### **输出**

```
MIMIC3_BENCHMARK_PATH
├── output_mimic3/                          # 预处理输出数据
│   ├── pheno/                              # 任务：表型分类
│   │   ├── ts_train.pkl                    # 规则时间序列数据
│   │   ├── norm_ts_train.pkl               # 不规则时间序列数据
│   │   ├── mean_std.pkl                    # 均值和标准差
│   │   ├── train_p2x_data.pkl              # 多模态训练数据
│   │   └── ...(val/test)
│   ├── ihm/                                # 任务：住院死亡率预测
│   └── readm/                              # 任务：30天再入院预测
```

- 原始的时间序列 `pkl` 文件：
  - `ts_train.pkl`、`ts_val.pkl`、`ts_test.pkl`：保存规则化后的时间序列数据。
  
- norm后的时间序列：
  - `norm_ts_train.pkl`、`norm_ts_val.pkl`、`norm_ts_test.pkl` 等，包含了不规则时间序列数据及掩码矩阵。
  
- norm时的均值与标准差：
  - `mean_std.pkl`：保存每个特征的均值和标准差。
  
- 多模态数据：
  - `train_p2x_data.pkl`、`val_p2x_data.pkl`、`test_p2x_data.pkl`：包含norm后时间序列和文本数据的多模态数据集。
  
  - pkl文件存了所有train/val/test 样本
  
  - **每个样本i的关键字段**:
  
    - `reg_ts`: **norm后的**规则时间序列数据，形状 `(48, 34)`，表示 48 个时间步、34 个变量(前17个为features，后17个为feature masks)。
  
    - `name`: str 样本名称，例如 `'367_episode1_timeseries.csv'`。
  
    - `label`: 目标标签，表示任务的监督信号（如住院死亡率）。
  
    - `ts_tt`: 不规则时间序列的时间戳列表（以小时为单位)，形状为`(T_i,)`。eg.[0.19, 0.32, 0.34, 1.19, 2.19]
  
      - 表示 ICU 住院过程中，病人发生事件的时间（单位：小时）。
      - 这些时间点不一定均匀分布。
  
    - `irg_ts`: **norm后**的不规则时间序列数据，形状为`(T_i, 17)`。
  
    - `irg_ts_mask`: 不规则时间序列的掩码信息，形状为`(T_i, 17)`。
  
    - `text_data`: List[str] 包含文本信息，表示临床笔记内容。
  
    - `text_time`: List[float] 包含文本时间信息，表示文本笔记相对于 ICU 入院时间的时间点。
  
      \- 表示距离ICU入院时间的小时数
  
      \- 用于对齐时间序列和文本数据
  
    - example：
  
      ```
      {'reg_ts': array([[-0.04923362, -0.05541235, -0.46559095, ...,  0.        ,
               0.        ,  0.        ],
             [-0.04923362, -0.08234494, -0.46559095, ...,  1.        ,
               0.        ,  0.        ],
             [-0.04923362, -0.07336741, -0.46559095, ...,  0.        ,
               0.        ,  1.        ],
             ...,
             [-0.04923362, -0.06438988, -0.46559095, ...,  0.        ,
               0.        ,  0.        ],
             [-0.04923362, -0.06438988, -0.46559095, ...,  0.        ,
               0.        ,  0.        ],
             [-0.04923362, -0.06438988, -0.46559095, ...,  0.        ,
               0.        ,  0.        ]]), 
          'name': '10011_episode1_timeseries.csv', 
          'label': 1, 
          'ts_tt': [0.43, 1.43, 2.43, 2.93, 3.43, 4.43, 5.43, 5.85, 6.43, 7.43, 8.43, 9.43, 10.43, 11.43, 12.43, 13.43, 14.43, 15.43, 16.43, 17.43, 18.43, 19.43, 20.43, 21.43, 22.43, 23.43, 24.43, 25.43, 26.43, 27.43, 27.77, 28.43, 29.43, 30.43, 31.43, 32.43, 33.43, 34.43, 35.43, 36.43, 37.43, 38.43, 39.43, 40.43, 41.43, 42.43, 43.43, 44.43, 46.43], 
          'irg_ts': array([[ 0.00000000e+00, -5.20557325e-02,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00, -2.89958544e-01,
               0.00000000e+00, -1.15742986e-01,  1.92339049e-03,
               4.33164878e-04, -5.49662337e-01,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00],
             
             [ 0.00000000e+00, -4.34559922e-02,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00,  5.72955569e-02,
               0.00000000e+00, -8.88931160e-02,  1.92339049e-03,
              -3.13245333e-03, -4.18140143e-01, -8.22416591e-04,
               0.00000000e+00,  0.00000000e+00],
             [ 0.00000000e+00, -4.34559922e-02,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00,  7.51803758e-01,
               0.00000000e+00, -8.88931160e-02,  1.92339049e-03,
               3.69850775e-05, -4.18140143e-01,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00],
             [ 0.00000000e+00, -4.77558623e-02,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00, -4.38781730e-01,
               0.00000000e+00, -1.02368332e-01,  1.92339049e-03,
              -3.92481293e-03, -4.83901240e-01,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00],
             [ 0.00000000e+00, -5.20557325e-02,  0.00000000e+00,
               6.72751595e-01, -2.05647542e-01,  3.55471276e-01,
               4.35662981e-01,  0.00000000e+00, -4.88389458e-01,
               0.00000000e+00, -1.35855249e-01, -1.87935301e-03,
              -2.73627353e-03, -7.46945628e-01,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00],
             [ 0.00000000e+00, -5.20557325e-02,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00, -6.37212644e-01,
               0.00000000e+00, -9.23122006e-02,  1.92339049e-03,
              -3.52863313e-03, -3.19498498e-01,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00],
             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
               0.00000000e+00, -2.68901235e-02,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00],
             [ 0.00000000e+00, -3.05563817e-02,  0.00000000e+00,
              -3.26903522e-01, -2.05647542e-01, -1.55104550e-01,
              -9.09706759e-02, -3.04975548e-02, -9.15276291e-02,
               0.00000000e+00, -4.20315445e-02,  1.92339049e-03,
              -2.34009373e-03, -1.55095755e-01,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00],
             [ 0.00000000e+00, -3.05563817e-02,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00, -6.86820373e-01,
               0.00000000e+00, -5.87247224e-02,  2.20187381e-05,
              -3.13245333e-03, -3.19498498e-01,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00],
             [ 0.00000000e+00,  2.53419305e-02,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00, -1.13328993e+00,
               0.00000000e+00,  5.85297676e-02,  9.72704612e-04,
              -3.92481293e-03, -2.35735610e-02,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00],
             [ 0.00000000e+00, -2.19566413e-02,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00, -1.03407447e+00,
               0.00000000e+00, -7.21999382e-02,  1.92339049e-03,
              -3.52863313e-03, -5.82542886e-01,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00],
             [ 0.00000000e+00, -2.19566413e-02,  0.00000000e+00,
              -3.26903522e-01, -2.05647542e-01,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00, -5.87604916e-01,
               0.00000000e+00, -5.54061991e-02,  1.92339049e-03,
              -2.73627353e-03, -4.18140143e-01, -7.94699918e-02,
               0.00000000e+00,  0.00000000e+00],
             [ 0.00000000e+00, -3.91561220e-02,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00, -9.84466745e-01,
               0.00000000e+00, -7.55184615e-02,  1.92339049e-03,
              -3.52863313e-03, -3.52379046e-01,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00],
             [ 0.00000000e+00, -9.05703081e-03,  0.00000000e+00,
              -3.26903522e-01,  5.05437928e-01,  3.55471276e-01,
               4.35662981e-01,  0.00000000e+00, -8.35643559e-01,
               0.00000000e+00, -5.12554301e-03,  0.00000000e+00,
              -3.13245333e-03, -1.22215207e-01,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00],
             [ 0.00000000e+00, -3.05563817e-02,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00, -9.15276291e-02,
               0.00000000e+00, -4.20315445e-02,  1.92339049e-03,
              -2.73627353e-03, -1.55095755e-01,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00],
             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00,  5.72955569e-02,
               0.00000000e+00,  0.00000000e+00,  2.20187381e-05,
              -2.34009373e-03,  0.00000000e+00,  0.00000000e+00,
               0.00000000e+00,  0.00000000e+00]]), 
          'irg_ts_mask': array([[0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
             [0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
             [0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
             [0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
             [0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0],
             [0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0]]), 
          'text_data': ["nursing note : admission day : 2230-0300 nkda wt 57.7 kg pt is a 36 yr old female who was transferred from for ? liver transplant . she was dx with hep b & c in . pt was recently admitted from at . she was actively using iv heroine until when she entered detox and was placed on methadone . on pt went to the ew with c/o sweats , n/v and was mildly jaundiced . at this time her lft 's were very high ( alt 1078 ast 1045 , inr 1.5 ) ruq ultrasound showed a normal liver and normal bile duct , no ascites . pt was treated with vit k and methadone . on she began lamivudine 100mg for hep b tx . later her inr rose to 3.3 and was transferred with ffpx 2 . her highest inr got to 5.0 . pmh : hep b & c heroin withdrawal with needle sharing and unprotected sex ptsd , anemia , reflux , migraines , skull fx , cocaine use . neg for hiv social pt lives with mother and has 4 children this admission mother states pt had changes in mental status and brought her into the ew .", "nursing note micu a 2230-0700 neuro : pt lethargic , is aware of name , does not know the year , when asked she states . she knows that she is in the hospital , but not the name even after several times of re orientating pt . pearl 2mm and sluggish , sclera jaundiced . speech is clear , but pt slow to answer . hand grasps are strong and equal . as well as foot pushes . pt is able to lift arms up and can hold for the count of 15 , same for foot raises . pt denies any pain or discomfort at this time . resp : bs clear . r/a sat 's 97-100 % abg 's this am . 7.47/31/119/23 no resp distress notes . cv : sbp 109-110 hr 72-84 nrs no ectopy noted.t max 97.5 compression boots on . skin : color is jaundiced , lower legs and upper arms pt has noteable track marks . sm bruise noted on lower left leg , no active bleeding noted . gu/gi : abd soft and slightly distended . on arrival to micu , pt had charcol stools x 2 . none since 0500. quiac neg . righ upper quad slightly painful on palp . positive bs x 4 . pt has foley to cd draining golden color urine 50-60cc per hr . urine sample and culture sent to lab this am . acess : right single lumen ij patent and intact . # 22 qc in left hand . iv d5.45ns at 100c per hr via perfi site . plan ? liver team to transplant pt .", '12:36 pm chest ( portable ap ) clip # reason : baseline cxr , transplant w/u medical condition : 36 year old woman with fulminant hepatic failure transferred from osh for possible liver transplant . reason for this examination : baseline cxr , transplant w/u final report indication : evaluation for liver transplant in patient with hepatic failure . portable chest : the cardiac and mediastinal contours are normal . there is no pulmonary vascular congestion , parenchymal consolidation or pleural effusion . the skeletal structures are unremarkable . there is no prior study for comparison . impression : no evidence of cardiopulmonary disease .', '3:03 pm us abd limit , single organ ; duplex dopp abd/pel clip # reason : hep b/c fulminant hepatic failure pre liver tx eval medical condition : 36 year old woman with reason for this examination : 36 y.o . female with hepb/c admitted with fulminant hepatic failure . possible candidate for liver transplant . please r/o pv thrombosis , hepatic vein thrombosis , ascites . final report indications : 36-year-old female with hep b/c admitted with fulminant hepatic failure , possible candidate for liver transplant . please evaluate for portal vein thrombosis , hepatic vein thrombosis , ascites . comparisons : none . findings : the liver is normal in size without evidence of focal mass . no ascites is demonstrated . the portal vein is widely patent with antegrade flow throughout . the hepatic venous system is also widely patent with no evidence of thrombus . the gallbladder is normal in size without evidence of stones , wall thickening , or pericholecystic fluid . there is no intra- or extrahepatic biliary dilatation . incidental note is made of a small polyp in the gallbladder wall . impression : 1 . no evidence of portal vein or hepatic vein thrombosis . 2 . no ascites . 3 . no focal masses in the liver . 4 . incidental note of gallbladder polyp .', '3:17 pm chest ( portable ap ) clip # reason : ngt placement medical condition : 36 year old woman with fulminant hepatic failure transferred from osh for possible liver transplant . reason for this examination : ngt placement final report indication : check tube placement . findings : a portable view of the lower chest and upper abdomen show the tip of a nasogastric tube in the stomach . the side port of the tube is a few cm below the diaphragm . the lung bases are clear . impression : tip of nasogastric tube in stomach .', "npn cv : vss , hr tachy to the low 100s at times . resp : ls clear , 02 sat high 90s on ra , rr in the teens gi : receiving lactulose , no stool , pos bs , lactulose increased to q2 hrs . gu : urine is amber colored , some sediment , u/o 40-50cc/hr . she conts on ivf at 100cc/hr , during the am she was very thirsty and had been drinking 200-400cc of gingerale at a time . neuro : her ms some during the day . her speach is a little slow - though increased from this morning , she is arousible to voice , knows her first name , knows that she is in a hospital , does not know the day or season - this am she knew her full name , knew that she was in a hospital , did not know the date . this afternoon she appears to be having a more difficult time concentrating , slow to follow commands and not following them consistantly - this am she was able to follow commands consistantly . pt has been quiet , staying in bed , no attemps to climb out . hepatology : seen by the liver transplant team , still being evaluated for a possible transplant . her inr , pt , ptt cont to increase , lfts will be checked tomorrow , ms is slowly deteriorating . soc : finally was able to get in touch with pt 's mother . she lives with her mother , has been an ivda since the age of 19 , has been in and out of rehabs numerous times , has been able to stay off of drugs for up to 6 months . she had 4 children all of whom she has lost to dss . her mother said that she would be in at 3 pm today and has not shown up , her boyfriend has been in most of the afternoon . neither nor her mother have a phone , we are only able to get in touch with her mother at work , a friend , or boyfriend pager .", '7:50 pm chest ( portable ap ) clip # reason : vomited this pm . please r/o aspiration medical condition : 36 year old woman with fulminant hepatic failure transferred from osh for possible liver transplant . reason for this examination : vomited this pm . please r/o aspiration final report clinical indication : rule out aspiration . portable ap chest dated is compared to the prior study performed 4 hours earlier . findings : there is a ngt terminating within the stomach . the cardiac and mediastinal contours are within normal limits . the pulmonary vascularity is unremarkable . the lungs appear clear , with no focal areas of opacification . there are no pleural effusions . impression : no evidence of aspiration .', 'micu npn 7a-7p stable overnoc . review of systems neuro : perrl . frequent neuro checks . pt continues to be oriented to 1st name only . pt had one episode of clarity where she discussed her 4 children . rare movement on bed noted . pt able to wiggle toes during the period of clarity . otherwise no movement noted . resp : ra . sats 97-100 % cv : hr nsr 64-89 . no ectopy noted . sbp stable 97-115/49-54 . k this am 2.9 . 40 meq kcl given po . 40 kcl meq hanging iv . gi : @ midnoc pt had minimal stool . ms worsening . rectal foley placed and lactulose enema given . pt stool 400cc in bag and then stooled around rectal foley . rectal foley d/ced . pt stooled ~400cc outside of bag . black liquid stool . guiac neg . abdomen soft , non tender gu : uo good 40-120cc/hr . bun 26 . creatinine 3 this am social : fiance called with page # . he would like to be called if she deteriorates . heme : hct 32.3 this am ( stable ) . pt 37.3 and inr 9.7 this am dispo : remains in micu . full code .', 'npn 7p-7a addendum : bun 3 and creatinine .6 .', 'npn vs : bp atable , hr increasing from the 80s to low 100s through the day . resp : ls clear , 02 sat high 90s on ra , rr in the teens gi : had a lg bm this am after q1 hr 30cc po lactulose , pt vomited at noon so was no longer given po and has received a lactulose enema . ms did improve this am after the frequent po doses , this afternoon she is much more somulent . gu : pt temporarily lost her iv access due to infiltration - she now has a # 20 in her r anticube . when she was not receiving ivf due to the access problem her u/o to ~ 20cc/hrs , it has increased since she is now receiving her ivf neuro : she was very somulent this am , unable to tell me her name or follow any commands , by noon she could tell me her full name , knew what hospital she was in , followed all commands . this afternoon she is again very somulent , only moaning to pain , not verbalizing - given a lactulose enema . soc : her boyfriend was in to visit , he was told of her poor prognosis and was upset by this . her mother has not been in , we have tried to contact her so we can get permission to given her ffp and place a cl ; we have not been successful in getting in touch with her . hepatology : ptt 9.0 at 9 am , resent this afternoon , k being repleated , sma 10 sent this afternoon as well . seen by liver service this am , felt that she has acute b hep on chronic hep c based on her ivda hx , because of this she is not that likely to have high intracranial pressure so a bolt was not placed . they also said that due to her social hx she is not that likely to be able to receive a liver transplant , she may however be able to have a hematocyte transplant .', '7:35 pm chest ( single view ) port clip # reason : ej placement assess position needed stat clinical information & questions to be answered : final report indication : history of hepatitis b and c. now with fulminant hepatic failure . evaluation of right external jugular cvl placement . comparison : portable ap chest : new right neck cvl has tip in right atrium . ng tube has tip and side port below the diaphragm . cardiac , mediastinal and hilar contours are stable . pulmonary vascularity is normal . the lung appear clear , with no pneumothorax . there are no pleural effusions . the soft tissue and osseous structures are unremarkable . impression : new right neck cvl with tip in right atrium . no pneumothorax . no other change as compared to prior study .'],
          'text_time': [4.483333333333333, 7.816666666666666, 14.033333333333333, 16.483333333333334, 16.716666666666665, 18.616666666666667, 21.266666666666666, 31.983333333333334, 32.583333333333336, 42.0, 45.016666666666666]}
      """
      ```
      
      

> **`reg_ts` vs. `irg_ts` 的区别**
>
> | 名称              | 形状          | 内容                                |
> | ----------------- | ------------- | ----------------------------------- |
> | **`reg_ts`**      | `(B, 48, 2C)` | **规则时间序列**，包括原始值 + mask |
> | **`irg_ts`**      | `(B, T_i, C)` | **不规则时间序列**，只有原始值      |
> | **`irg_ts_mask`** | `(B, T_i, C)` | **不规则时间序列的 mask**           |
>
> **irgts (irregular time series):**
>
> 是原始的、不规则采样的时间序列数据
>
> 特点:
>
> 采样时间点不固定
>
> 每个时间点可能缺失某些特征值
>
> 包含mask标记缺失值
>
> **regts (regular time series):**
>
> 是通过discretizer处理后的规则采样时间序列
>
> 特点:
>
> 固定时间间隔采样(由args.timestep指定)
>
> 使用插值方法(如previous)填充缺失值
>
> 长度固定(由args.period_length指定)
>
> **两者的关系:**
>
> irgts是原始数据
>
> regts是通过对irgts进行重采样和插值得到的规则时间序列
>
> 在模型中会同时使用这两种表示,并通过attention机制进行融合



> 处理完的pkl多模态文件：
>
> # MIMIC 数据集详细说明（适配TIME-LLM模型）
>
> 下面是针对处理好的 MIMIC 数据集的详细字段说明、数据含义、shape及具体示例信息，供coding参考：
>
> ## 数据集整体结构
>
> 数据集是一个列表，每个元素代表一个患者或一次入院记录（episode）。  
> 每个样本为一个字典，包含结构化多元时间序列数据、非结构化临床文本数据及其对应时间戳、任务标签等字段。
>
> ## 样本详细描述 (字段说明)
>
> | 字段名 | 描述 | 数据类型 | Shape / 长度示例 | 含义说明 |
> |---|---|---|---|
> | `reg_ts` | `numpy.ndarray` | `(24, 34)` | 规则化后的多元时间序列数据，每小时采样一次（含插值后值），共24小时，34个特征维度（生命体征、检验指标等）|
> | `irg_ts` | `numpy.ndarray` | `(24, 17)` | 原始不规则采样的多元时间序列数据，存在缺失值，17个特征维度|
> | `irg_ts_mask` | `numpy.ndarray` | `(24, 17)` | 与 `irg_ts` 一一对应的缺失值掩码 (0表示缺失，1表示存在) |
> | `ts_tt` | `list[float]` | 长度为`24` | 时间序列的时间戳 (单位: 小时)，表示irg_ts每个测量值的时间点 |
> | `text_data` | `list[str]` | 例如长度为`3` (每个样本可能不同) | 非结构化的临床文本记录列表 (护理记录、医生病程记录等) |
> | `text_time` | `list[float]` | 与`text_data`长度相同 | 临床文本记录对应的时间戳 (单位: 小时) |
> | `label` | `list[int]` | 例如长度为`26` | 任务标签，例如48小时内院内死亡 (0或1)，或多标签分类（25个常见病症）。注意首个元素可能是患者标识符 |
> | `name` | `str` | 例如 `'19781_episode1_timeseries.csv'` | 样本唯一标识 |
>
> ---
>
> 建议在复现timellm时只使用reg_ts，text_data，text_time，label，name
>
> ## 多元时间序列特征说明（共17个，`irg_ts`字段示例）：
>
> 时间序列特征包含以下临床指标：
>
> 1. Capillary refill rate
> 2. Diastolic blood pressure
> 3. Fraction inspired oxygen
> 4. Glascow coma scale eye opening
> 5. Glascow coma scale motor response
> 6. Glascow coma scale total
> 7. Glascow coma scale verbal response
> 8. Glucose
> 9. Heart Rate
> 10. Height
> 11. Mean blood pressure
> 12. Oxygen saturation
> 13. Respiratory rate
> 13. Systolic blood pressure
> 14. Temperature
> 15. Weight
> 16. pH
> 17. (可能存在其他指标，总共17维，如数据所示)
>
> ---
>
> ## 形式化定义
>
> 定义数据样本为：
>
> 令第$i$个样本为字典形式，记为：
>
> $
> X^{(i)} = \{ \text{reg\_ts}, \text{irg\_ts}, \text{irg\_ts\_mask}, \text{ts\_tt}, \text{text\_data}, \text{text\_time}, \text{label}, \text{name}\}$
>
> **结构化多元时序数据 (规则化后)**:
> $\text{reg\_ts}^{(i)} \in \mathbb{R}^{T \times D},\quad T=24,\quad D=34$
>
> - 其中$T$为采样时间步数（每小时1步，共24小时）。
> - $D$为临床特征维度数量（34个特征，包括生命体征如心率、血压、体温、血氧饱和度等）。
>
> - 具体临床特征举例如下：
>   $\text{Heart Rate, Mean BP, Oxygen saturation, Respiratory rate, Temperature, Glucose, ..., pH}$
>
> 非结构化文本数据定义为：
>
> $
> \text{text\_data}^{(i)} = [\text{text}_1, \text{text}_2, \dots, \text{text}_M],\quad M \text{ 为文本笔记数量, 每条笔记为不定长字符串}$
>
> 对应的文本时间戳：
>
> $
> \text{text\_time}^{(i)} = [t_1, t_2, \dots, t_M], \quad \text{其中每个值表示笔记记录时间点（单位小时）}
> $
>
> - 示例:
>   - 文本内容：
>     ```markdown
>     [
>       "ccu nsg admit note-micu (...) 80 yo female admitted ...",
>       "9:25 am chest (portable ap) clip # reason: assess for rij ..."
>     ]
>     
>     

### Usage

Please see the scripts in `scripts/mimic3`.

#### To run CTPD approach:

`cd scripts/mimimc3`

##### 住院死亡率预测任务

run`sh train_ctpd_ihm.sh`

##### 表型分类任务

run`sh train_ctpd_pheno.sh`



他们都会运行`train_mimic3.py` 使用不同model对不同任务进行train/test

**2 main components**：

**data**

```
dm = MIMIC3DataModule(
            file_path=str(
                DATA_PATH / f"output_mimic3/{args.task}"), # PreProcess 好的dataset路径，pkl文件
            tt_max=args.period_length,
            batch_size=args.batch_size,
            modeltype="TS_Text",
            num_workers=args.num_workers,
            first_nrows=args.first_nrows)
```

**model**

```
model = CTPDModule.load_from_checkpoint(
                    args.ckpt_path, **vars(args))
```



##### `train_mimic3.py` 进行**训练和测试的完整流程**

1. 训练过程

   - 训练过程中，每个`epoch`结束后：
     - 计算 `val_auprc`
     - 如果 **比之前最好的 `val_auprc` 更优**，则 **保存 checkpoint**
     - 如果 **连续 5 轮 `val_auprc` 没有提升**，则 **提前终止**
     - 如果 **达到了 `max_epochs`，则停止训练**

2. 测试过程

   ```
   trainer.test(model, datamodule=dm, ckpt_path="best")
   ```

   - 选取 **`val_auprc` 最高的 checkpoint** 进行测试
   - 计算 `test_auprc`, `test_auroc`, `test_f1` 等指标
   - 记录最终结果

#### MIMIC3DataModule

##### 整体结构概览

1. **TSNote_Irg(Dataset)**
   - 这是一个自定义的 PyTorch `Dataset` 类，用来读取预处理好的数据（pkl文件），并将每个样本里包含的“不规则时间序列 + 规则时间序列 + 文本”整理成可训练所需的张量数据。
   - 内部最核心的函数是 `__getitem__` 和 `__len__`。
   - `__getitem__`：根据样本索引，读取对应的多模态 EHR 数据，并进行必要的编码与填充操作，最终返回统一格式的字典数据。
   - `__len__`：返回数据条目的总数。
2. **TextTSIrgcollate_fn**
   - 这是一个自定义的 `collate_fn` 函数，用于在 `DataLoader` 批处理（batch）时，对同一个 batch 里的样本进行打包和对齐（pad）。
   - 最终返回一个包含所有重要键（ts、ts_mask、ts_tt、reg_ts、input_ids、attention_mask、note_time、note_time_mask、label）的大字典。
3. **MIMIC3DataModule(LightningDataModule)**
   - 这是一个结合了 PyTorch Lightning 的数据模块，方便在训练、验证、测试阶段分别构造不同的 `DataLoader`。
   - 内部会分别实例化 `TSNote_Irg`（Dataset），并指定相应的 `collate_fn`。

下面将详细分模块说明，穿插注释它在多模态医疗预测任务中的用法与形状信息。

##### TSNote_Irg(Dataset) 类

```
class TSNote_Irg(Dataset):
    def __init__(self,
                 file_path: str,
                 split: str,
                 bert_type: str,
                 max_length: int,
                 modeltype: str = "TS_Text",
                 nodes_order: str = "Last",
                 num_of_notes: int = 5,
                 tt_max: int = 48,
                 first_nrows: Optional[int] = None):
        super().__init__()

        data_path = os.path.join(file_path, f"{split}_p2x_data.pkl")
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
```

1. `file_path`：数据所在的根目录路径。
2. `split`：表示当前数据集是哪一部分（train/val/test）。
3. `bert_type`：指定所用的文本编码器（如 "prajjwal1/bert-tiny" 或 "yikuan8/Clinical-Longformer"）。
4. `max_length`：文本分词后最大长度限制，用于后续 tokenizer 的 `encode_plus`。
5. `modeltype`：控制模型类型，默认为 "TS_Text" 表示既需要时间序列也需要文本，如果是单独时间序列模型可能不需要文本处理。
6. `nodes_order`：表示对于文本序列，是取靠前的几条还是靠后的几条。可设成 "Last" 或 "First"。
7. `num_of_notes`：固定要取的临床文本条数，例如 5 条。
8. `tt_max`：在做时间编码（time to X）时的一个归一化上限，比如 48 小时。
9. `first_nrows`：可选，用来只取前几条数据做测试或调试。
10. `self.data`：从 `pkl` 文件中加载出来的 List，每个元素是一个 dict，对应一条病人样本的多模态信息。

```
if split == "train":
    self.notes_order = nodes_order
else:
    self.notes_order = "Last"
```

- 如果是训练集，可以通过 `notes_order` 决定取靠前文本还是靠后文本（可做数据增强）。验证、测试则固定为 “Last”。

```
ratio_notes_order = None
if ratio_notes_order != None:
    self.order_sample = np.random.binomial(
        1, ratio_notes_order, len(self.data))
```

- 这里写死了 `ratio_notes_order=None`，所以暂时不会触发随机二项分布的逻辑。如要动态地随机取文本起始顺序，可以自己设置 ratio_notes_order。

```
self.modeltype = modeltype
self.bert_type = bert_type
self.num_of_notes = num_of_notes
self.tt_max = tt_max
self.max_len = max_length
self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
self.first_nrows = first_nrows
```

- 初始化一些在后续处理中要用到的超参数。
- `self.tokenizer`：HuggingFace 的分词器，用于对文本做分词、编码。

```
if self.first_nrows != None:
    self.data = self.data[:self.first_nrows]
```

- 如果只想取数据的前几行做实验，就在这一步切片。

```
print("Number of original samples: ", len(self.data))
self.data = list(filter(lambda x: len(x['irg_ts']) < 1000, self.data))
print(f"Number of filtered samples in {split} set: {len(self.data)}")
```

- 只保留不规则时间序列（`irg_ts`）长度小于 1000 的样本，以去除太长的异常样本。
- `irg_ts` 一般是形如 (T_i, D) 的二维结构，这里 T_i < 1000 以限制极值。



###### `__getitem__`

这是数据读取最核心的函数，根据索引获取一条样本并返回。

```
def __getitem__(self, idx):
    if self.notes_order != None:
        notes_order = self.notes_order
    else:
        notes_order = 'Last' if self.order_sample[idx] == 1 else 'First'

    data_detail = self.data[idx]
    idx = data_detail['name']
    reg_ts = data_detail['reg_ts']  # shape: (48, 34)
    ts = data_detail['irg_ts']      # shape: (T_i, 17)
    ts_mask = data_detail['irg_ts_mask']  # shape: (T_i, 17)
```

1. `notes_order`：决定取靠前还是靠后的文本序列。

2. `data_detail`：当前索引对应的一个病人记录，是一个 `dict`

   

   示例：

   ```
   {
       'name': '病人id',
       'reg_ts': array形状(48,34),
       'irg_ts': array形状(T_i,17),
       'irg_ts_mask': array形状(T_i,17),
       'text_data': [...(若干条文本)...],
       'label': int 或者 multi-label,
       'ts_tt': [...], 
       'text_time': [...]
       # 以及其他key...
   }
   ```

3. `idx`：这里又把 `data_detail['name']` 赋值给 `idx`，当做后续输出的标识符。

4. `reg_ts`：规则时间序列，形状为 `(48, 34)`。这里 48 大概率是将 ICU 最多 48 小时按小时离散化出来，每小时 34 个生理指标。

5. `ts` 和 `ts_mask`：不规则时间序列和对应的掩码，形状分别是 `(T_i, 17)`。`T_i` 表示该病人的不规则观测总数，17 是特征维度。`ts_mask` 里一般是 0/1 或者其他标签。

```
if 'text_data' not in data_detail:
        return None
```

- 若无文本数据则返回 `None`，在后面 `collate_fn` 会过滤掉。

```
  text = data_detail['text_data']
  if len(text) == 0:
      return None

  text_token = []
  atten_mask = []
  label = data_detail["label"]
  ts_tt = data_detail["ts_tt"]
  text_time = data_detail["text_time"]
```

- `text`：该样本对应的临床文本列表，比如 `[ "病人xx...", "病人xx...", ... ]`。
- `label`：标签，用于后续监督训练。如二分类则是 0/1，或者多类别时是 int。
- `ts_tt`：不规则时间点的时间戳（或者距离结局的小时数），后续会做归一化。
- `text_time`：对应文本发生的时间戳/时间信息，同样会做归一化。

###### 文本处理

```
if 'Text' in self.modeltype:
    for t in text:
        inputs = self.tokenizer.encode_plus(
            t,
            padding="max_length",
            max_length=self.max_len,
            add_special_tokens=True,
            return_attention_mask=True,
            truncation=True
        )
        text_token.append(torch.tensor(inputs['input_ids'], dtype=torch.long))
        attention_mask = inputs['attention_mask']
        if "Longformer" in self.bert_type:
            attention_mask[0] += 1
            atten_mask.append(torch.tensor(attention_mask, dtype=torch.long))
        else:
            atten_mask.append(torch.tensor(attention_mask, dtype=torch.long))
```

1. 循环遍历每条文本`t`，用 tokenizer 编码：
   - `input_ids`：形如 `[101, 7592, ... , 102, 0, 0]` 的分词 ID，长度固定为 `max_length`。
   - `attention_mask`：相应的注意力掩码，为 1/0，标明哪些是 padding。
   - `token_type_ids`（若使用 BERT）等信息同样会在返回的 `inputs` dict 里，但这里没使用或只默认。
2. 把 `input_ids` 和 `attention_mask` 转成 PyTorch `tensor` 存到 `text_token`、`atten_mask`。
3. 形状：
   - `text_token[i]` -> `(max_length,)`
   - `atten_mask[i]` -> `(max_length,)`

```
label = torch.tensor(label, dtype=torch.long)
```

- `label` 转为长整型张量 `(1,)` 或者标量张量。

###### 时间序列处理

```
reg_ts = torch.tensor(reg_ts, dtype=torch.float)  # shape: (48, 34)
ts = torch.tensor(ts, dtype=torch.float)          # shape: (T_i, 17)
ts_mask = torch.tensor(ts_mask, dtype=torch.long) # shape: (T_i, 17)

ts_tt = torch.tensor([t/self.tt_max for t in ts_tt], dtype=torch.float) 
text_time = [t/self.tt_max for t in text_time]
text_time_mask = [1] * len(text_time)
```

- `reg_ts`、`ts`、`ts_mask` 转为张量。
- 将 `ts_tt` 和 `text_time` 做归一化，即除以 `tt_max`（默认为 48）。它们是不规则时间点/文本时间的相对刻度。
- `text_time_mask`：长度与文本数量相同，全部置为1，用于标识这些文本time的存在。

```
if 'Text' in self.modeltype:
    # 如果文本数量不足 num_of_notes，就补足
    while len(text_token) < self.num_of_notes:
        text_token.append(torch.tensor([0], dtype=torch.long))
        atten_mask.append(torch.tensor([0], dtype=torch.long))
        text_time.append(0)
        text_time_mask.append(0)

text_time = torch.tensor(text_time, dtype=torch.float)
text_time_mask = torch.tensor(text_time_mask, dtype=torch.long)
```

- 如果文本数量少于 `num_of_notes`（比如只采集到 3 条，但设定要 5 条），这里用 0 padding 做填充。
- `text_time_mask` 也同步写 0。
- 最终 `text_time` 和 `text_time_mask` 都是长度 `num_of_notes` 的一维向量。

###### 返回字典

```
if 'Text' not in self.modeltype:
    return {'idx': idx, 'ts': ts, 'ts_mask': ts_mask,
            'ts_tt': ts_tt, 'reg_ts': reg_ts,
            "label": label}

if notes_order == "Last":
    return {
        'idx': idx,
        'ts': ts,
        'ts_mask': ts_mask,
        'ts_tt': ts_tt,
        'reg_ts': reg_ts,
        "input_ids": text_token[-self.num_of_notes:],
        "label": label,
        "attention_mask": atten_mask[-self.num_of_notes:],
        'note_time': text_time[-self.num_of_notes:],
        'text_time_mask': text_time_mask[-self.num_of_notes:]
    }
else:
    return {
        'idx': idx,
        'ts': ts,
        'ts_mask': ts_mask,
        'ts_tt': ts_tt,
        'reg_ts': reg_ts,
        "input_ids": text_token[:self.num_of_notes],
        "label": label,
        "attention_mask": atten_mask[:self.num_of_notes],
        'note_time': text_time[:self.num_of_notes],
        'text_time_mask': text_time_mask[:self.num_of_notes]
    }
```

- 若只训练时间序列，直接返回 ts、reg_ts 等；若也需要文本，就根据 `notes_order` 返回靠后的 `num_of_notes` 条或者靠前的 `num_of_notes` 条。
- 返回的字典各键最终形状示例（以文本数为num_of_notes=5,max_length=512为例）：
  - `ts`：`(T_i, 17)`
  - `ts_mask`：`(T_i, 17)`
  - `ts_tt`：`(T_i,)`
  - `reg_ts`：`(48, 34)`
  - `input_ids`：list 长度 5，每个元素 `(max_length,)`
  - `attention_mask`：list 长度 5，每个元素 `(max_length,)`
  - `note_time`：`(5,)`
  - `text_time_mask`：`(5,)`
  - `label`：标量或 `(1,)`

###### `__len__`

```
def __len__(self):
    return len(self.data)
```

- 数据集中样本数量。

##### TextTSIrgcollate_fn

```
def TextTSIrgcollate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    batch = list(filter(lambda x: len(x['ts']) < 1000, batch))  # type: ignore
    if len(batch) == 0:
        return None
```

- 过滤掉为空的项（或 ts 过长的），若都为空则直接返回 None。

```

ts_input_sequences = pad_sequence(
    [example['ts'] for example in batch], 
    batch_first=True, 
    padding_value=0
) 
ts_mask_sequences = pad_sequence(
    [example['ts_mask'] for example in batch], 
    batch_first=True, 
    padding_value=0
)
ts_tt = pad_sequence(
    [example['ts_tt'] for example in batch], 
    batch_first=True, 
    padding_value=0
)
label = torch.stack([example["label"] for example in batch])
```

- `pad_sequence`：对可变长的不规则序列在时间维度做对齐，以 `[B, T_max, D]` 形式堆叠。这里对齐是为了确保一个batch内的数据长度一样
- `ts_input_sequences` 形状：`(batch_size, max_T_i, 17)`
- `ts_mask_sequences` 形状：`(batch_size, max_T_i, 17)`
- `ts_tt` 形状：`(batch_size, max_T_i)`
- `label` 一般是 `(batch_size,)`

```
reg_ts_input = torch.stack([example['reg_ts'] for example in batch])
```

- `reg_ts_input`：形状 `(batch_size, 48, 34)`。

```
		if len(batch[0]) > 6:  # 判断是否有文本相关字段
        input_ids = [pad_sequence(example['input_ids'], batch_first=True,
                                  padding_value=0).transpose(0, 1) for example in batch]
        attn_mask = [pad_sequence(example['attention_mask'], batch_first=True,
                                  padding_value=0).transpose(0, 1) for example in batch]

        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=0).transpose(1, 2)
        attn_mask = pad_sequence(
            attn_mask, batch_first=True, padding_value=0).transpose(1, 2)

        note_time = pad_sequence([example['note_time'].clone().detach().float()
                                  for example in batch], batch_first=True, padding_value=0)
        note_time_mask = pad_sequence([example['text_time_mask'].clone().detach().long()
                                      for example in batch], batch_first=True, padding_value=0)
    else:
        input_ids, attn_mask, note_time, note_time_mask = None, None, None, None
```

1. `input_ids` 和 `attention_mask`
   - 每个 `example['input_ids']` 本身是个长度 `num_of_notes` 的 list，每个元素形状 `(max_length,)`。
   - `pad_sequence(..., batch_first=True)` 会变成 `(num_of_notes, max_length)`。
   - 再 `.transpose(0,1)` 得 `(max_length, num_of_notes)`。
   - 外面再对整个 batch 做 `pad_sequence`，最后 `.transpose(1,2)`，把 `batch` 维度、`max_length` 维度和 `num_of_notes` 维度对齐到一个固定形状。
   - 最终形状通常是 `(batch_size, num_of_notes, max_length)`。
   - 具体步骤稍显复杂，主要原因是每个病人可能有不同数量的临床文本、且每条文本分词后都需要 pad，所以要多重 pad + transpose。
2. `note_time`：对每个样本不同时刻的 note_time 做 pad，最后形状 `(batch_size, max_num_of_notes)`。
3. `note_time_mask`：同理 `(batch_size, max_num_of_notes)`。

```
    return {
        "ts": ts_input_sequences,
        "ts_mask": ts_mask_sequences,
        "ts_tt": ts_tt,
        "reg_ts": reg_ts_input,
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "note_time": note_time,
        "note_time_mask": note_time_mask,
        "label": label
    }
```

- 最终返回一个大字典，整合了所有多模态对齐后的批量数据。
- 例如：
  - `ts` => `[B, T_max, 17]`
  - `ts_mask` => `[B, T_max, 17]`
  - `reg_ts` => `[B, 48, 34]`
  - `input_ids` => `[B, num_of_notes, max_length]`（如果包含文本）
  - `attention_mask` => `[B, num_of_notes, max_length]`
  - `note_time` => `[B, num_of_notes]`
  - `text_time_mask` => `[B, num_of_notes]`
  - `label` => `[B]`

##### MIMIC3DataModule(LightningDataModule)

最后这个类主要是 PyTorch Lightning 常用的数据模块写法，它会分别返回训练、验证和测试的 DataLoader。

```
class MIMIC3DataModule(LightningDataModule):
    def __init__(self,
                 batch_size: int = 4,
                 num_workers: int = 1,
                 modeltype: str = "TS_Text",
                 file_path: str = str(ROOT_PATH / "output/ihm"),
                 bert_type: str = "prajjwal1/bert-tiny",
                 max_length: int = 512, 
                 tt_max: int = 48,
                 first_nrows: Optional[int] = None) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.file_path = file_path
        self.bert_type = bert_type
        self.max_length = max_length
        self.modeltype = modeltype
        self.first_nrows = first_nrows
        self.tt_max = tt_max
```

1. `batch_size`：一批数据量。
2. `num_workers`：DataLoader 进程数。
3. `modeltype, file_path, bert_type, max_length, tt_max, first_nrows` 同前。

然后其下有三个函数：

- `train_dataloader`: 用 `TSNote_Irg` 构造训练集 dataset，指定 `split="train"`，然后包一层 `DataLoader`。
- `val_dataloader`: 构造验证集，`split="val"`。
- `test_dataloader`: 构造测试集，`split="test"`。

所有 DataLoader 都使用 `TextTSIrgcollate_fn` 作为 `collate_fn`，以便批量拼接。

##### 代码在多模态预测（如 CTPD）中的作用

这段数据类与 DataModule 的组合，在多模态 EHR（包含文本与时间序列）的深度学习任务中非常常见。和论文 [CTPD: Cross-Modal Temporal Pattern Discovery for Enhanced Multimodal Electronic Health Records Analysis] 里的思路对应起来，这里的不规则时间序列（`irg_ts`、`ts_tt`）及规则时间序列（`reg_ts`）可以被编码成时序特征，而临床文本则需要用 `BERT` 或类似的 Transformer 来提取文本表征。
在 **CTPD** 这样的多模态框架中，往往需要：

1. 将时间序列处理成隐藏向量（可以包含对不规则性处理或原生 48 小时序列处理），如论文中提到的多尺度时序注意力、插值等做法。
2. 将文本处理成固定长度或可变长度的 token embedding，通过“最后 K 条或最前 K 条”等方式获取临床文本对病情演化的描述。
3. 在模型内部再做时序特征与文本特征的融合（如论文提到的交叉模态注意力、对齐等），以更好地捕捉跨模态时序模式，提升临床任务预测效果。

本代码完成了把各模态数据从原始 pkl 数据整理成可训练张量的步骤：

- 分别对齐并 padding。
- 对文本分词与注意力掩码。
- 对不规则时间序列 mask 并 padding。
- 统一输出到 batch 里。

如此，后续任意的多模态模型（包括 CTPD 等）都可以直接从 `DataLoader` 里拿到形如 `batch["ts"]`、`batch["input_ids"]`、`batch["reg_ts"]` 等键进行训练。

##### 变量与张量形状总结

- **reg_ts**: `(48, 34)`，规则时间序列，48 个离散时间点 × 34 维特征。
- **irg_ts (ts)**: `(T_i, 17)`，不规则时间序列，T_i 条观测 × 17 维特征，每个样本长度不同。
- **irg_ts_mask (ts_mask)**: `(T_i, 17)`，同尺寸的掩码，标识缺失或已观测位置。
- **ts_tt**: `(T_i,)`，不规则观测在时间轴上的位置，用于做时间特征（除以 tt_max 归一化）。
- **text_data**: `List[str]`，若干条文本。如果要经过 tokenizer，最终 `(num_of_notes, max_length)`；若文本少于 `num_of_notes`，会 0 pad 补齐。
- **attention_mask**: `(num_of_notes, max_length)`，对应文本的注意力掩码。
- **text_time**: `(num_of_notes,)`，每条文本对应的时间戳（归一化）。
- **text_time_mask**: `(num_of_notes,)`，标识文本时间有效性。
- **label**: 标量或单维 `(1,)`，分类或回归标签。

在 `collate_fn` 处理之后，batch 级别会多一层 batch 维度，变成：

- `ts` => `[batch_size, max_T_i, 17]`
- `input_ids` => `[batch_size, num_of_notes, max_length]`
- 等等。

##### 每一步操作的意义

1. **过滤/选择样本**：在 Dataset 初始化时先过滤掉 `len(irg_ts) >= 1000` 的极长序列，这样可以避免过长异常数据对训练的干扰。
2. **文本编码**：对 `text_data` 做分词、截断或填充至 `max_length`，并且构建注意力掩码，以便后续 Transformer 编码。
3. **对不规则时间序列做张量转换**：把 `irg_ts`（含 mask）转为 float/long 张量；此外对时间 `ts_tt` 做归一化，便于后续模型使用相对时间信息（如论文里做时序插值、跨模态注意力等）。
4. **保持固定文本数**：通过 “Last/First” 逻辑取最后几条或最前几条文本，若不足 `num_of_notes` 则用 0 填充，以保证 batch 里每个样本的文本数是一致的（便于并行训练）。
5. **collate_fn 做批对齐**：将不同样本的时序数据、文本数据等进行 pad，对齐到相同时间长度或文本数量，形成批量大张量，以供后续 GPU 并行计算。
6. **LightningDataModule**：方便管理训练、验证、测试三个阶段的数据加载，对每个阶段都使用相同的 Dataset 类和 collate_fn，只是 `split` 不同，自动读取相应 pkl 文件并返回 DataLoader。

##### 总结

本代码是一个典型的多模态医疗数据读取和对齐流程，用来把“不规则和规则时间序列 + 临床文本”融合到同一个 batch 中，为下游的多模态模型（如 CTPD 或其他深度学习模型）提供整洁且一致的输入。通过这些预处理与填充操作，就能够让模型在训练阶段更专注于学习跨模态时序模式，而不必处理各种变长与缺失细节。

- **重点变量**：
  - `ts`, `ts_mask`, `reg_ts`：表征了多维生理指标的时序数据（不规则 + 规则）。
  - `text_data` -> `input_ids`, `attention_mask` -> `(num_of_notes, max_length)`：表征了多条临床文本经过分词后的张量。
  - `ts_tt`, `text_time`：时间信息做归一化后，为模型提供时点或时间差，便于捕捉演化轨迹。
  - `label`：监督学习标签。
- **形状**：代码中对不规则时序与文本均需要 padding；输出的 batch 里，每个键都会增加 batch 维度。
- **与 CTPD 的关联**：此类多模态数据处理正是论文所述“跨模态时序分析”的前提。CTPD 中需对时间序列与文本同时进行建模，代码里做了文本 tokenizer、时间序列张量化与归一化，并保证可在 batch 里对齐，这在论文提出的融合与时序模式挖掘中非常关键。

通过这套流程，就能把原始 `.pkl` 里的多模态数据，统一封装成可被下游深度学习模型直接读取的批量张量形式，为多模态 EHR 的模型训练（如 CTPD）打下基础。



> torch.nn.utils.rnn.pad_sequence是什么，为什么需要在这里被使用？
>
> 在 PyTorch 中，`pad_sequence` 是一个用于对可变长度序列（variable-length sequences）进行对齐（padding）的实用函数，来自 `torch.nn.utils.rnn` 模块。它能够将一组不同长度的张量（例如一批句子或一批时间序列）沿着指定的维度补齐到同一长度，然后打包成一个统一尺寸的张量，方便后续在神经网络中并行处理。
>
> **背景与作用**
>
> 1. **可变长度序列**
>    在很多任务中（如自然语言处理、时间序列分析等），每个样本的序列长度往往不同。例如，某些病人只有 3 条文本，而另一些病人有 5 条文本；或不同时长的时间序列观测数 T 不一样。要把它们组成一个 batch（批量）在 GPU 上并行运算，就需要先将它们扩充到相同长度。
> 2. **对齐（Padding）**
>    `pad_sequence` 就是用来对张量序列进行“补零”或其他指定的 `padding_value` 填充，使得每条序列在被堆叠成一个张量时大小一致。例如，如果一批句子最长分词长度是 10，那么对于长度不足 10 的句子，末尾补齐到 10；对于时间序列也一样，可以在时间维度上做补齐。
> 3. **batch_first 参数**
>    当我们在 `pad_sequence` 里指定 `batch_first=True` 时，返回的张量就会是 `[batch_size, max_seq_len, ...]` 这种格式。这样后续在模型（如 RNN、Transformer）中做并行计算非常方便。







#### CTPD model

> ##### 主要模块与相互关系
>
> - **`SlotAttention` 类**
>   - 用于学习一组“Slot”表示，可理解为论文中“Prototype”或“原型模式”的实现，提取输入序列中的关键模式。
>   - 对应论文中 “Prototype Discovery” 的思想，通过迭代式注意力 + GRU + MLP 的方式来更新 slot。
>   - 其 forward 输入是 `slots` 和 `inputs`，输出是更新后的 `slots` 以及 slot 对输入的注意力分配 `attn`。
> - **`CTPDModule` 类**
>   - 继承自 `MIMIC3LightningModule`（这是一个基于 PyTorch Lightning 的封装）。
>   - 主要包含：
>     1. **多模态输入的编码**（如不规则时间序列 mTAND、规则时间序列的一维卷积、文本 BERT 编码 + mTAND）；
>     2. **多尺度特征提取**（`ts_conv_1`, `ts_conv_2`, `ts_conv_3` 对时间序列进行多尺度卷积）；
>     3. **SlotAttention 模块**（如果 `use_prototype` 为 True，则会对时序、文本分别使用 SlotAttention 获取原型表示并做跨模态对比）；
>     4. **解码器部分（用于重构损失）**（`ts_decoder` / `text_decoder`）；
>     5. **最终特征融合**（`fusion_layer` 以及后续的 pooling / out_layer）。
>     6. **损失函数**（交叉熵 + 对比损失 + 重构损失）。
> - **`multiTimeAttention`**
>   - 来自于 `cmehr.models.mimic4.UTDE_modules`，实现了与论文类似的多时间注意力机制 (mTAND)。
>   - 主要将不规则时间序列 / 文本序列根据时间戳映射到统一的参考时间线上。
> - **训练流程**
>   - 从数据读取：不规则时间序列 (irregular TS), 规则时间序列 (regular TS), 文本 (token + attention mask) 等；
>   - 使用 mTAND 提取特征：
>     - 不规则时序: `forward_ts_mtand`
>     - 规则时序: `forward_ts_reg`
>     - 文本: `forward_text_mtand`
>   - 若开启 `TS_mixup`，将不规则/规则时序进行门控融合 (`gate_ts`)。
>   - 多尺度卷积提取时序特征 (`ts_conv_1,2,3`)。
>   - 文本做一层卷积得到文本特征。
>   - 如果 `use_prototype`，则使用 `SlotAttention` 获取时序 / 文本的原型表示，并计算跨模态对比损失与重构损失。
>   - 融合（`fusion_layer`）得到最终特征做分类回归（`out_layer`）。
>   - 计算 loss: 包括 交叉熵损失 + 对比损失 (TPNCE) + 重构损失。

##### `SlotAttention` 类

**模块/函数名称**: `SlotAttention`

- 这是一个提取“原型槽 (slots)”的注意力模块，灵感来自于图像领域的 Slot Attention (Locatello et al. 2020) 等类似方法。这里被用在时序/文本上，用迭代的注意力和 GRUCell 来更新 slots。

**变量解析**:

- `dim`: 槽（slot）向量的维度，和输入 embedding 维度一致。
- `iters`: 迭代次数 (默认为3)，表示多次循环更新 slots 的过程。
- `eps`: 数值稳定用的小偏移量。
- `scale`: 缩放系数 `dim**-0.5`，在计算点乘注意力时使用。
- `to_q`, `to_k`, `to_v`: 将输入特征映射为 query/key/value 的线性层。
- `gru`: GRUCell，用于迭代更新每个 slot 的隐藏状态。
- `mlp`: 一个两层 MLP，用于在迭代更新后再做一次前馈更新
- `norm_input`, `norm_slots`, `norm_pre_ff`: LayerNorm，用于在更新时做归一化操作，帮助训练稳定。

**核心逻辑**: 3次迭代更新
每一次迭代 (共 iters 次)：

1. 对 `slots` 做 LayerNorm，作为 q；对输入的TS 或者 TEXT embedding做线性得到 k, v；
2. 计算注意力分数 `dots = q * k^T * scale`，之后 softmax 并规范化；
3. 用注意力权重来对 v 求加权和作为 `updates`；
4. 用 GRUCell 更新 slots `slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d))`
5. 再加上一个 MLP 残差连接`slots = slots + self.mlp(self.norm_pre_ff(slots))`。

最终输出的 `slots` 就是迭代收敛后的原型槽表示。

<img src="/Users/haochengyang/Desktop/Research/CTPD/MMMSPG-014C/IMG_95BDC9DBD9E4-1.jpeg" alt="IMG_95BDC9DBD9E4-1" style="zoom:25%;" />

**数据流向**:

- `slots`: `[B, num_slots, D]`
- `inputs`: `[B, T, D]` (T 表示 time或序列长度)
- 输出: `slots, attn`，其中 `attn` 是slots对TS/TEXT的注意力分配情况 `[B, num_slots, T]`。













#### To run baselines:

```
sh ts_baseines.sh
sh note_baselines.sh
```





### Experiment

需要手动设置是否use_prototype

通过运行脚本中的`--lamb1 0.1 --lamb2 0.5 --lamb3 0.5` 进行Ablation Studies on Learning Objectives







# Appendix

#### **1. Time series features selection and extraction**

> Link: https://github.com/YerevaNN/mimic3-benchmarks/blob/master/README.md
>
> 其中`python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/`**将每个患者的数据**分割成多个 **ICU 住院期间的 episode**。例如，一个患者在 ICU 住院 3 次，最终会生成 **3 个 episode**：
>
> - `episode1_timeseries.csv`，`episode2_timeseries.csv`，`episode3_timeseries.csv`
>
> **最终处理后的文件结构**：
>
> ```
> data/
> ├── root/                              # 原始数据按患者存储
> │   ├── 10001/                         # 患者10001的数据
> │   │   ├── stays.csv                  # ICU停留信息（INTIME、OUTTIME、LOS等）
> │   │   ├── diagnoses.csv              # 诊断信息（ICD代码、疾病描述等）
> │   │   ├── events.csv                 # 原始事件数据（时间、ITEMID、数值）
> │   │   ├── episode1.csv               # 第1次ICU住院的汇总信息（年龄、性别、住院时长、死亡率）
> │   │   ├── episode1_timeseries.csv    # 第1次ICU住院的时间序列数据（按小时采样的生理指标）
> │   │   ├── episode2.csv               # 第2次ICU住院的汇总信息
> │   │   └── episode2_timeseries.csv    # 第2次ICU住院的时间序列数据
> │   ├── 10002/
> │   │   ├── stays.csv
> │   │   ├── diagnoses.csv
> │   │   ├── events.csv
> │   │   ├── episode1.csv
> │   │   └── episode1_timeseries.csv
> │   └── ...
> │
> ├── in-hospital-mortality/             # 任务1：住院死亡率预测
> │   ├── train/                         # 训练集
> │   │   ├── 10001_episode1_timeseries.csv
> │   │   ├── 10001_episode2_timeseries.csv
> │   │   ├── 10002_episode1_timeseries.csv
> │   │   └── ...
> │   ├── test/                          # 测试集
> │   │   ├── 10003_episode1_timeseries.csv
> │   │   ├── 10004_episode1_timeseries.csv
> │   │   └── ...
> │   └── listfile.csv                   # 包含train和test样本列表（每行包含: ICU_ID, 时长, 标签）
> │
> ├── decompensation/                    # 任务2：失代偿预测
> │   ├── train/
> │   ├── test/
> │   └── listfile.csv
> 。。。
> 
> ```
>
> 在 `data/root/` 目录下，每个患者（以 `SUBJECT_ID` 命名的文件夹）包含以下文件：
>
> - `stays.csv`：记录 ICU 停留信息。
> - `diagnoses.csv`：记录诊断信息。
> - `events.csv`：记录所有事件。
> - `episode{#}.csv`：每次 ICU 停留的汇总信息（如患者年龄、性别、种族、身高、体重、死亡率、住院时间、诊断等）。
> - `episode{#}_timeseries.csv`：每次 ICU 停留期间的时间序列数据。
> - ==一个episode对应一个HADM_ID==
>
> 在 `data/` 目录下，每个baseline任务（如 `in-hospital-mortality`、`decompensation`、`length-of-stay`、`phenotyping`）都有一个独立的文件夹，包含：
>
> - `train/`：训练集数据。
> - `test/`：测试集数据。
> - `listfile.csv`：列出训练或测试集中包含的样本。
>
> 每个 `train/` 和 `test/` 文件夹包含多个以 `{SUBJECT_ID}_episode{#}_timeseries.csv` 命名的文件，对应特定患者的特定 ICU 停留的时间序列数据。

#### 2. Clinical notes extraction

> [!IMPORTANT]
>
> **SUBJECT_ID与HADM_ID**
>
> `SUBJECT_ID`：患者唯一标识，**每个患者唯一**
>
> `HADM_ID`：住院唯一标识，**每次住院唯一**
>
> 一个 `SUBJECT_ID` 可对应**多个** `HADM_ID`

> Link: https://github.com/kaggarwal/ClinicalNotesICU
>
> - **从 MIMIC-III 的 `NOTEEVENTS.csv` 提取临床笔记**
>
> - **清洗并分割文本**，
>
> - **将每位患者的笔记数据按 ICU 住院（`HADM_ID`）分组**，
>
> - **保存为 JSON 文件**，供后续训练使用。

##### 输入

`NOTEEVENTS.csv`包含 MIMIC-III 中所有患者的临床笔记，包括：

- **SUBJECT_ID**（患者ID）
- **HADM_ID**（住院ID）
- **CHARTTIME**（记录时间）
- **TEXT**（笔记内容）

##### 输出

- JSON文件（如`10001_1.json`）：存储了按时间戳排序的分词后的笔记，并以 **`HADM_ID`** 进行区分。
- `test_hadm_id2index`: 一个 JSON 文件，存储了 `HADM_ID` 到索引的映射。

##### 生成的文件树

```
/disk1/*/EHR_dataset/mimiciii_benchmark/
├── train/                                  # 原始训练数据
│   ├── 10001/                              # 患者10001
│   │   ├── stays.csv                       # ICU住院信息
│   │   ├── diagnoses.csv                   # 诊断信息
│   │   ├── events.csv                      # 事件数据
│   │   ├── episode1_timeseries.csv         # 时间序列数据
│   │   └── ...
│   └── ...
│
└── train_text_fixed/                       # 输出目录，存放生成的JSON文件
    ├── 10001_1.json                        # 患者10001第1次住院的笔记数据
    ├── 10001_2.json                        # 患者10001第2次住院的笔记数据
    ├── 10002_1.json                        # 患者10002第1次住院的笔记数据
    ├── ...                                 # 其他患者的笔记JSON文件
    └── test_hadm_id2index                  # HADM_ID到索引的映射

```

##### **`test_hadm_id2index` example**

```json
{
  "2001": "0",
  "3001": "1",
  "4001": "2"
}
```

- `HADM_ID` → 索引：
  - `2001` → `0`
  - `3001` → `1`
  - **便于后续训练快速索引数据**。

> [!IMPORTANT]
>
> **关于索引**
>
> **`HADM_ID` 到索引的映射**
>
> - 在预处理过程中，每个 ICU 住院都有一个唯一的 **`HADM_ID`**。
> - 代码生成了一个 **`索引`**，用来表示每个 **`HADM_ID`**，便于在训练时快速加载对应的文本数据
>
> **为什么需要索引？**
>
> - 索引是为了方便在训练时快速访问数据。
>
> - 在训练过程中，模型会读取 `HADM_ID`，并通过索引快速找到对应的 笔记 JSON 数据







#### 3. Multimodal Mimic

> Link: https://github.com/XZhang97666/MultimodalMIMIC?tab=readme-ov-file









Yes, your summary is correct. For each sample $i$, the key fields can be interpreted as follows:

1. **$\text{reg\_ts} \,(48, 34)$**  
   - This is the **regular (grid-aligned) time series**, shaped $(48, 34)$.  
   - The first 17 columns represent actual physiological features (e.g., blood pressure, heart rate, etc.), while the next 17 columns are the corresponding masks (0/1) indicating whether each feature is missing or present at each time step.  
   - Typically, $48$ represents hourly sampling for the first 48 hours of ICU stay; $34$ corresponds to $17$ features plus $17$ masks.

2. **$\text{name} \,(\text{str})$**  
   - A string identifier for the sample, e.g., `"367_episode1_timeseries.csv"`.  
   - Used to track the original file or patient episode.

3. **$\text{label}$**  
   - The supervision label for the prediction task (e.g., in-hospital mortality or phenotype classification).  
   - It could be binary $(0/1)$ or a multi-label vector, depending on the task.

4. **$\text{ts\_tt} \,(T_i)$**  
   - A list of timestamps for the irregular time series (in hours), of length $T_i$.  
   - For example, $[0.19, 0.32, 1.19, 2.19, \dots]$ indicates the specific hours at which observations were recorded.  
   - The total number of recorded observations $T_i$ varies per patient.

5. **$\text{irg\_ts} \,(T_i, 17)$**  
   - The **irregular time series** data, shape $(T_i, 17)$.  
   - Each row corresponds to one observation timestamp, with 17 physiological features (after normalization or other preprocessing).

6. **$\text{irg\_ts\_mask} \,(T_i, 17)$**  
   - A **mask matrix** for the irregular time series, also shape $(T_i, 17)$.  
   - Each element is $1$ if that feature is observed at the corresponding time, and $0$ if it is missing.

7. **$\text{text\_data}$**: List of strings  
   - A list of clinical notes, each element is a segment of free-text medical notes for this patient.

8. **$\text{text\_time}$**: List of floats  
   - A list of timestamps (in hours) corresponding to each entry in $\text{text\_data}$.  
   - For example, $[1.2, 4.0, 10.5]$ means these notes were recorded $1.2$ hours, $4$ hours, and $10.5$ hours after ICU admission.  
   - This aligns the textual notes with the time-series data on a shared timeline.

Overall, these fields capture both **regular** and **irregular** time series representations of a patient (including feature values and masks), as well as **clinical text** entries with their timestamps, plus a **label** for the supervised prediction task.







我们有一个有限集合里面含有N 个可能的对象，也有它们的先验分布。GIven一个提前指定的的对象of interest，我们希望通过最短的一系列 “是/否”（Yes/No）问题，来唯一确定这个对象是有限集合中的哪一个。







归根结底，只要题目能抽象为：

- 有一批可能对象及其先验概率，
- 需用二分问题把**唯一**目标对象精确识别出来，

