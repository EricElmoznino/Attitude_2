import Attitude_1
import Helpers

att_model = Attitude_1.Model(Helpers.Configuration(epochs=40), 100, 100)
# att_model.train('/Users/Eric/ML_data/Attitude_1/data_pitch/train_data',
#                 '/Users/Eric/ML_data/Attitude_1/data_pitch/validation_data',
#                 '/Users/Eric/ML_data/Attitude_1/data_pitch/test_data')
# predictions = att_model.predict('/Users/Eric/ML_data/Attitude_1/data_pitch/prediction_data')
att_model.train('./data/train_data',
                './data/validation_data',
                './data/test_data')
predictions = att_model.predict('./data/prediction_data')
print(predictions)
