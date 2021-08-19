from text_params_ import *
from text_data import *
from text_model import *
from text_eval import evaluate_model

X, sum_X = gen_text_train_data()

deep_we = train_model(X, sum_X, get_deepset_model, models_layer_arr['deepset'])
lstm_we = train_model(X, sum_X, get_lstm_model,  models_layer_arr['lstm'])
gru_we = train_model(X, sum_X, get_gru_model,  models_layer_arr['gru'])

print(evaluate_model(get_deepset_model, deep_we))
print(evaluate_model(get_lstm_model, lstm_we))
print(evaluate_model(get_gru_model, gru_we))

