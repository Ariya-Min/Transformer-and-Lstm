import numpy as np 
import pandas as pd
from torch import nn
import  torch
def cor(X,Y):
	# 手写的计算相关系数的cor函数
	X,Y=np.array(X),np.array(Y)
	X,Y=X-np.mean(X),Y-np.mean(Y)
	numerator=np.dot(X,Y)
	denominator=(np.dot(X,X)*np.dot(Y,Y))**0.5
	return round(numerator/denominator,6)

def Get_data():
	the_df=pd.read_csv("industry_ret_data_1500.csv").set_index("Date").iloc[20:,:].fillna(0)

	# 对每一天的所有行业的ret做demean操作，对Moving average来说做或不做不影响，对其它predictor可能有影响
	for i in range(the_df.shape[0]):
		the_df.iloc[i,:]=the_df.iloc[i,:]-the_df.iloc[i,:].mean(0)
	return the_df

def Simple_predictor():
	# 最简单的predictor，求20天moving average，再向后shift1格，表示用t-19到t天的moving average预测t+1天
	The_df=Get_data()
	print(The_df.shape)
	Predicted_signal=The_df.rolling(window=20).mean().shift(1)
	Predicted_signal.fillna(0,inplace=True)
	# print(Predicted_signal)
	return Predicted_signal
criterion = nn.MSELoss()
def Compute_information_coefficient(The_df,Predicted_signal):
	# 因为取了前20天做MA因此前20天的数据丢掉
	The_df=The_df.iloc[20:,:]
	Predicted_signal=Predicted_signal.iloc[20:,:]
	#
	for day in The_df.index:
		print(day,cor(The_df.loc[day,:],Predicted_signal.loc[day,:]))
	IC_list=[cor(The_df.loc[day,:],Predicted_signal.loc[day,:]) for day in The_df.index]
	IC_dict={"Date":The_df.index,"IC":IC_list}
	IC_df=pd.DataFrame(IC_dict)
	IC_df.to_csv("IC.csv")
	# The_df=The_df.astype('float32')
	# Predicted_signal=Predicted_signal.astype('float32')
	# print(Predicted_signal.shape)
    #print(torch.tensor(np.array(Predicted_signal)).size(),torch.tensor(np.array(The_df)).size())
	#print(torch.tensor(np.array(Predicted_signal)).size(),torch.tensor(np.array(The_df)).size())
	M_Loss=criterion(torch.tensor(np.array(Predicted_signal)),torch.tensor(np.array(The_df)))
	print("mse:",M_Loss)
	print(np.mean(IC_list))

def Test():
	
	The_df=Get_data()
	Predicted_signal=Simple_predictor()
	print(Predicted_signal)
	Compute_information_coefficient(The_df,Predicted_signal)

if __name__ == '__main__':
	Test()
