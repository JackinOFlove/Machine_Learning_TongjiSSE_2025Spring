from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('Stroke_predict.csv')

# 提取预测值和实际值
predictions = df['predictTime']
actuals = df['factTime']

# 计算MAE
mae = mean_absolute_error(actuals, predictions)

# 计算RMSE
rmse = np.sqrt(mean_squared_error(actuals, predictions))

print(f'MAE: {mae}')
print(f'RMSE: {rmse}')
# 计算实际值的平均值
mean_actuals = np.mean(actuals)

# 计算相对 MAE
relative_mae = mae / mean_actuals

# 计算相对 RMSE
relative_rmse = rmse / mean_actuals

print(f'Relative MAE: {relative_mae}')
print(f'Relative RMSE: {relative_rmse}')
