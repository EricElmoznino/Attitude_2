import Attitude_2
import Helpers

att_model = Attitude_2.Model(Helpers.Configuration(epochs=40), 100, 100)
# att_model.train('/Users/Eric/ML_data/Attitude_2/train_data',
#                 '/Users/Eric/ML_data/Attitude_2/validation_data',
#                 '/Users/Eric/ML_data/Attitude_2/test_data')
# predictions = att_model.predict('/Users/Eric/ML_data/Attitude_2/prediction_data')
att_model.train('./data/train_data',
                './data/validation_data',
                './data/test_data')
predictions = att_model.predict('./data/prediction_data')
print(predictions)
