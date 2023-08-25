import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SEQ_LEN = 60 # mins. how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 3 # how far into the future are we trying to predict?
RATIO_TO_PREDICT = "ETH-USD"
EPOCHS = 10
BATCH_SIZE = 64
NAME = f"{RATIO_TO_PREDICT}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"

def classify(current, future):  # current price, future price
	if float(future) > float(current):
		return 1
	else:
		return 0

def preprocess_df(df):
	#df = df.drop("future", 1)

	for col in df.columns: # go through all of the columns
		if col != "target": # normalize all ... except for the target itself!
			df[col] = df[col].pct_change() # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
			df.dropna(inplace = True) # remove the nas created by pct_change
			df[col] = preprocessing.scale(df[col].values) # scale between 0 and 1.

	df.dropna(inplace=True) # cleanup again

	sequential_data = [] # this is a list that will CONTAIN the sequences
	prev_days = deque(maxlen=SEQ_LEN) # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in
	
	for i in df.values: # iterate over the values
		prev_days.append([n for n in i[:-1]]) # store all but the target
		if len(prev_days) == SEQ_LEN: # make sure to have 60 sequences
			sequential_data.append([np.array(prev_days), i[-1]]) # append

	random.shuffle(sequential_data) # shuffle for good measure.

	buys = []
	sells = []

	for seq, target in sequential_data:
		if target == 0:
			sells.append([seq, target])
		elif target == 1:
			buys.append([seq, target])

	random.shuffle(buys)
	random.shuffle(sells)

	lower = min(len(buys), len(sells))

	buys = buys[:lower]
	sells = sells[:lower]

	sequential_data = buys+sells
	random.shuffle(sequential_data)

	X = []
	y = []

	for seq, target in sequential_data:
		X.append(seq)
		y.append(target)

	return np.array(X), y


# df = pd.read_csv("crypto_data/LTC-USD.csv", names=["time", "low", "high", "open", "close", "volume"])

main_df = pd.DataFrame() # begin empty

ratios = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]  # the 4 ratios we want to consider
for ratio in ratios: # begin iteration
	#print(ratio)
	dataset = f"crypto_data/{ratio}.csv" # get the full path to the file.

	df = pd.read_csv(dataset, names=["time", "low", "high", "open", "close", "volume"])  # read in specific file
	#print(df.head())
	# rename volume and close to include the ticker so we can still which close/volume is which:
	df.rename(columns={"close":f"{ratio}_close", "volume":f"{ratio}_volume"}, inplace = True) # inplace to not have to redefine the df

	df.set_index("time", inplace = True) # set time as index so we can join them on this shared time
	df = df[[f"{ratio}_close", f"{ratio}_volume"]] # ignore the other columns besides price and volume


	if len(main_df)==0: # if the dataframe is empty
		main_df=df # then it's just the current df
	else: # otherwise, join this data to the main one
		main_df = main_df.join(df)

main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
main_df.dropna(inplace=True)
#print(main_df.head())  # how did we do??

main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"]))

#print(main_df[[f"{RATIO_TO_PREDICT}_close", "future", "target"]].head(10))


times = sorted(main_df.index.values)
last_5pct = times[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), activation="relu", return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:]), activation="relu", return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128, input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',
			optimizer=opt,
			metrics=['accuracy'])

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

filepath = "RNN_Final-{epoch:02d}-{val_accuracy:.3f}" # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
validation_x = np.asarray(validation_x)
validation_y = np.asarray(validation_y)

history = model.fit(
	train_x, train_y,
	batch_size=BATCH_SIZE,
	epochs=EPOCHS,
	validation_data=(validation_x, validation_y),
	callbacks=[tensorboard, checkpoint])

