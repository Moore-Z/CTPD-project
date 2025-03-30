import pandas as pd

# 读取csv文件
df = pd.read_csv('/Users/haochengyang/Desktop/research/CTPD/MMMSPG-014C/EHR_dataset/mimiciii_benchmark/ihm/train_listfile.csv')

# 假设最后一列为label
label_col = df.columns[-1]

# 统计0和1的数量
counts = df[label_col].value_counts()

# 计算比例
ratios = counts / counts.sum()

print("Label的数量统计：")
print(counts)

print("\nLabel的比例统计：")
print(ratios)
