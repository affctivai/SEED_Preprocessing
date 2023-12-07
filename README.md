# SEED_Preprocessing
raw data preprocessing for SEED_FRA and SEED_GER

SEED_FRA 와 SEED_GER .cnt파일로 된 raw data를 전처리하였습니다.

"M1","M2","VOE","MOE" 가 제외되었으며

0-75Hz로 Bandpass filter, Sampling Hertz는 200Hz로 Downsampling 해줍니다.

dataloader은 각 데이터셋을 Covariance 기반의 dataset으로 만들어줍니다.

혹시 궁금한점이나 이상한점이 있으면 말씀해주시면 감사하겠습니다
