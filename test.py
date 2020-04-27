from API.utils import *
import joblib
col_names = joblib.load('./data/phase_2/cleaned/col_names.pkl')
eva = EVA()

x4 = pd.read_csv('./data/phase_1/test_170.csv',index_col=0).sample(1)[['T10','T50','T90','N+A']].reset_index(drop=True)
case4 = pd.read_csv('./data/phase_2/cleaned/train.csv',index_col=0)[col_names['case']].sample(1).reset_index(drop=True)
Input = x4.join(case4)

print(eva(Input))

print(eva.predict)
print(eva.naphtha)
print(eva.pre_d)
print(eva.reform)