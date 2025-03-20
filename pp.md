

## BUG1：`listfile.csv` 不能直接适配 `InHospitalMortalityReader`，需要进行格式调整

`src/cmehr/preprocess/mimic3/mimic3benchmark/scripts/create_in_hospital_mortality.py` 生成的listfile.csv有4个columns:

<img src="/Users/haochengyang/.Trash/MMMSPG-014C/截屏2025-02-20 上午2.52.36.png" alt="截屏2025-02-20 上午2.52.36" style="zoom: 50%;" />

Example of csv file:

<img src="/Users/haochengyang/.Trash/MMMSPG-014C/截屏2025-02-20 上午2.54.45.png" alt="截屏2025-02-20 上午2.54.45" style="zoom: 33%;" />

而在运行`src/cmehr/preprocess/mimic3/mimic3models/create_iiregular_ts.py`时会调用`Class InHospitalMortalityReader`, 而他默认读取的listfile.csv只有2个columns，会产生下面的错误：

```
Traceback (most recent call last):
  File "/opt/miniconda3/envs/ctpd/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/opt/miniconda3/envs/ctpd/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/Users/haochengyang/Desktop/Research/CTPD/MMMSPG-014C/src/cmehr/preprocess/mimic3/mimic3models/create_iiregular_ts.py", line 517, in <module>
    create_irregular_ts()
  File "/Users/haochengyang/Desktop/Research/CTPD/MMMSPG-014C/src/cmehr/preprocess/mimic3/mimic3models/create_iiregular_ts.py", line 368, in create_irregular_ts
    train_reader = InHospitalMortalityReader(
  File "/Users/haochengyang/Desktop/Research/CTPD/MMMSPG-014C/src/cmehr/preprocess/mimic3/mimic3models/readers.py", line 107, in __init__
    self._data = [(x, int(y)) for (x, y) in self._data]
  File "/Users/haochengyang/Desktop/Research/CTPD/MMMSPG-014C/src/cmehr/preprocess/mimic3/mimic3models/readers.py", line 107, in <listcomp>
    self._data = [(x, int(y)) for (x, y) in self._data]
ValueError: too many values to unpack (expected 2)
```

`Class InHospitalMortalityReader`的代码：

<img src="/Users/haochengyang/.Trash/MMMSPG-014C/截屏2025-02-20 上午2.56.35.png" alt="截屏2025-02-20 上午2.56.35" style="zoom: 50%;" />

<img src="/Users/haochengyang/.Trash/MMMSPG-014C/截屏2025-02-20 上午2.58.18.png" alt="截屏2025-02-20 上午2.58.18" style="zoom: 50%;" />



### **✅ 方法 1：在 `InHospitalMortalityReader` 里修改 `_data` 解析**

**在 `InHospitalMortalityReader` 的 `__init__` 里修改 `_data` 解析方式：**

```
self._data = [line.split(',') for line in self._data]
self._data = [(x, int(y)) for (x, _, _, y) in self._data]  # 忽略 `period_length` 和 `stay_id`
```

📌 **这样 `InHospitalMortalityReader` 只保留 `stay` 和 `y_true`，就能适配版本 2 的 `listfile.csv`！**
