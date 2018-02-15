from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from os import path
import os, random

import OsuReader
import pyttanko
import subprocess
from scipy.io.wavfile import read

save_path = "D:\\Downloads\\BPM Identifier\\Saved models\\model.ckpt"

starter_learning_rate = 0.6
sampling_rate = 8000
audio_foresight = 1.2
audio_backsight = 0.2
audio_batch_length = audio_foresight + audio_backsight
audio_batch_size = int(sampling_rate * audio_batch_length) #must be int
place_range = 1000
num_epochs = 100
truncated_backprop_length = 60
state_size = 10
num_classes = 2
X_size = 3
batch_size = 1
audiostatestar = audio_batch_size + state_size + 1

def generateData():
    #get random beatmap dir
    folder = path.dirname(__file__)
    data_folder = path.join(folder, "Training Songs")
    beatmap_folder = path.join(data_folder, random.choice(os.listdir(data_folder)))
    beatmap_list = []
    for f in os.listdir(beatmap_folder):
        if f[-4:] == ".osu":
            beatmap_list.append(f)
    beatmap_path = path.join(beatmap_folder, random.choice(beatmap_list))
    #open the beatmap
    beatmap = OsuReader.readBeatmap(beatmap_path)
    p = pyttanko.parser()
    bmap = p.map(open(beatmap_path, 'rt', encoding="utf8"))
    #find the audio
    audio_path = path.join(beatmap_folder, beatmap.AudioFilename)
    #open the audio
    wav_path = path.join(beatmap_folder, "audio.wav.wav")
    if not path.exists(wav_path):
        subprocess.call(['ffmpeg', '-i', audio_path, "-ar", str(sampling_rate),
                         "-ac", "1",
                       path.join(beatmap_folder, "audio.wav.wav")])
    audio = read(wav_path)[1]
#    audio = np.divide(audio, np.full((len(audio)), 32767, np.float64), dtype=np.float64)
    #find the star rating
    difficulty = pyttanko.diff_calc().calc(bmap)
    stars = np.array(difficulty.total).reshape([1,1])

    print(beatmap.Title, beatmap.Version)

    #make list of all timings and delays between timings in beatmap
    delay_list = []
    timing_list = []
    current_time = 0
    for o in beatmap.HitObjects:
        if o.csstype == "circle":
            delay_list.append(o.time - current_time)
            current_time = o.time
            timing_list.append(current_time)
        elif o.csstype == "slider":
            delay_list.append(o.time - current_time)
            current_time = o.time
            timing_list.append(current_time)
            for t in range(o.repeat):
                delay_list.append(o.endTime)
                current_time += o.endTime
                timing_list.append(current_time)
        elif o.csstype == "spinner":
            delay_list.append(o.time - current_time)
            current_time = o.time
            timing_list.append(current_time)
            delay_list.append(o.endTime - current_time)
            current_time = o.endTime
            timing_list.append(current_time)

    #make x and y
    song_length = len(audio) / sampling_rate * 1000
    audio = np.concatenate((audio, np.zeros(audio_batch_size, dtype=np.float64)), axis=0)
    audio = np.concatenate((np.zeros(audio_batch_size, dtype=np.float64), audio), axis=0)
    current_time = 0
    current_timing = 0
    array_x = []
    array_y = []
    array_label = []
    while current_time < song_length:
        audio_index = int(current_time / 1000 * sampling_rate)
        if current_timing < len(timing_list):
            distance = timing_list[current_timing] - current_time
            if distance < place_range:
                array_label.append(np.array([1]))
                array_y.append(np.array(distance))
                array_x.append(audio[audio_index:audio_index+audio_batch_size])
                current_time = timing_list[current_timing]
                current_timing += 1
            else:
                array_label.append(np.array([0]))
                array_y.append(np.array(1000))
                array_x.append(audio[audio_index:audio_index+audio_batch_size])
                current_time += 1000
        else:
            array_label.append(np.array([0]))
            array_y.append(np.array(1000))
            array_x.append(audio[audio_index:audio_index+audio_batch_size])
            current_time += 1000

    while not len(array_x)%truncated_backprop_length == 0:
        array_label.append(np.array([0]))
        array_y.append(np.array(1000))
        array_x.append(np.zeros(audio_batch_size, dtype=np.float64))
    
    x = np.vstack(array_x)
    y = np.vstack(array_y)
    labels = np.vstack(array_label)
    
    return (x, y, stars, labels)



#this is the computational graph

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000, 0.96, staircase=False)

batchX_placeholder = tf.placeholder(tf.float64, [truncated_backprop_length, audio_batch_size])
batchY_placeholder = tf.placeholder(tf.float64, [truncated_backprop_length, 1])

stars_placeholder = tf.placeholder(tf.float64, [1,1])
labels_placeholder = tf.placeholder(tf.int64, [truncated_backprop_length, 1])

init_state = tf.placeholder(tf.float64, [state_size, 1])

# Variables
W1_1 = tf.Variable(np.random.rand(1, X_size), dtype=tf.float64)
W2_1 = tf.Variable(np.random.rand(1, X_size), dtype=tf.float64)
W3_1 = tf.Variable(np.random.rand(1, X_size), dtype=tf.float64)
W1_2 = tf.Variable(np.random.rand(audiostatestar, X_size), dtype=tf.float64)
W2_2 = tf.Variable(np.random.rand(audiostatestar, X_size), dtype=tf.float64)
W3_2 = tf.Variable(np.random.rand(audiostatestar, X_size), dtype=tf.float64)
W1_3 = tf.Variable(np.random.rand(X_size, state_size), dtype=tf.float64)
W2_3 = tf.Variable(np.random.rand(X_size, 1), dtype=tf.float64)
W3_3 = tf.Variable(np.random.rand(X_size, 1), dtype=tf.float64)
W1_4 = tf.Variable(np.zeros((X_size, 1)), dtype=tf.float64)
W2_4 = tf.Variable(np.random.rand(X_size, 1), dtype=tf.float64)
W3_4 = tf.Variable(np.random.rand(X_size, num_classes), dtype=tf.float64)
b1_1 = tf.Variable(np.zeros((audiostatestar, X_size)), dtype=tf.float64)
b2_1 = tf.Variable(np.zeros((audiostatestar, X_size)), dtype=tf.float64)
b3_1 = tf.Variable(np.zeros((audiostatestar, X_size)), dtype=tf.float64)
b1_2 = tf.Variable(np.zeros((X_size, X_size)), dtype=tf.float64)
b2_2 = tf.Variable(np.zeros((X_size, X_size)), dtype=tf.float64)
b3_2 = tf.Variable(np.zeros((X_size, X_size)), dtype=tf.float64)
b1_3 = tf.Variable(np.zeros((X_size, state_size)), dtype=tf.float64)
b2_3 = tf.Variable(np.zeros((X_size, 1)), dtype=tf.float64)
b3_3 = tf.Variable(np.zeros((X_size, 1)), dtype=tf.float64)
b1_4 = tf.Variable(np.zeros((state_size, 1)), dtype=tf.float64)
b2_4 = tf.Variable(np.zeros((1, 1)), dtype=tf.float64)
b3_4 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float64)

# Unpack rows
inputs_series = tf.unstack(batchX_placeholder, axis=0)
answers_series = tf.unstack(batchY_placeholder, axis=0)
labels_series = tf.unstack(labels_placeholder, axis=0)

# Forward pass
current_state = init_state
losses = []
answers_series = []
predictions_series = []
for current_input, current_answer, current_label in zip(inputs_series,answers_series,labels_series):
	current_input = tf.reshape(current_input, [audio_batch_size, 1])
	full_input = tf.concat([current_input, current_state, stars_placeholder], 0)
	full_input = tf.reshape(full_input, [audiostatestar, 1])
	#three different channels
	c1_1 = tf.matmul(full_input, W1_1) + b1_1
	c2_1 = tf.matmul(full_input, W2_1) + b2_1
	c3_1 = tf.matmul(full_input, W3_1) + b3_1
	c1_1 = tf.reshape(c1_1, [X_size, audiostatestar])
	c2_1 = tf.reshape(c2_1, [X_size, audiostatestar])
	c3_1 = tf.reshape(c3_1, [X_size, audiostatestar])
	c1_2 = tf.matmul(c1_1, W1_2) + b1_2
	c2_2 = tf.matmul(c2_1, W2_2) + b2_2
	c3_2 = tf.matmul(c3_1, W3_2) + b3_2
	next_state = tf.matmul(c1_2, W1_3) + b1_3
	next_state = tf.reshape(next_state, [state_size, X_size])
	next_state = tf.matmul(next_state, W1_4) + b1_4
	answer = tf.matmul(c2_2, W2_3) + b2_3
	answer = tf.reshape(answer, [1, X_size])
	answer = tf.matmul(answer, W2_4) + b2_4
	answer = tf.reshape(answer, [1])
	logits = tf.matmul(c3_2, W3_3) + b3_3
	logits = tf.reshape(logits, [1, X_size])
	logits = tf.matmul(logits, W3_4) + b3_4
	logits = tf.reshape(logits, [1, num_classes])

	logi_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=current_label)
	timing_loss = tf.abs(answer - current_answer)
	loss = tf.cond(tf.reshape(current_label, []) > 0, lambda: tf.add(timing_loss,logi_loss), lambda: logi_loss)
	losses.append(loss)
	
	current_state = next_state

	answers_series.append(answer)
	predictions_series.append(tf.nn.softmax(logits))
	

total_loss = tf.reduce_mean(losses)
total_loss = tf.clip_by_value(total_loss, 0, 1.7976931348623157e+307)

train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss, global_step=global_step)


def plot(loss_list, answers_series, batchY, predictions_series, batchLabels):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)
    
    plt.subplot(2,3,3)
    plt.cla()
    plt.axis([0, truncated_backprop_length, 0, place_range])
    plt.plot(np.array(answers_series).reshape((truncated_backprop_length)), color="green")
    plt.plot(batchY.reshape((truncated_backprop_length)), color="red")


    one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
    single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])
    
    plt.subplot(2, 3, 5)
    plt.cla()
    plt.axis([0, truncated_backprop_length, 0, 2])
    left_offset = range(truncated_backprop_length)
    plt.bar(left_offset, batchLabels[batch_series_idx, :] * 0.5, width=1, color="red")
    plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

##    for batch_series_idx in range(batch_size):
##        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
##        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])
##
##        plt.subplot(2, 3, batch_series_idx + 2)
##        plt.cla()
##        plt.axis([0, truncated_backprop_length, 0, 2])
##        left_offset = range(truncated_backprop_length)
##        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
##        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
##        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if input("Do you want to restore the model?(y/n): ") == "y":
        saver.restore(sess, save_path)
        print("Model restored.")
    
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        print("New data, epoch", epoch_idx)

        x,y,stars,labels = generateData()
        _current_state = np.zeros((state_size, 1))

        for batch_idx in range(len(x)//truncated_backprop_length):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length
            
            batchX = x[start_idx:end_idx,:]
            batchY = y[start_idx:end_idx,:]
            batchLabels = labels[start_idx:end_idx,:]


            _total_loss, _train_step, _current_state, _predictions_series, _answers_series, _learning_rate = sess.run(
                [total_loss, train_step, current_state, predictions_series, answers_series, learning_rate],
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                    stars_placeholder:stars,
                    labels_placeholder:batchLabels,
                    init_state:_current_state
                })

            loss_list.append(_total_loss)
            if len(loss_list) > 500:
                loss_list.pop(0)

            if batch_idx%(24*(epoch_idx>0)+1) == 0:
                print("Step",batch_idx, "Loss", _total_loss, "Leaning rate", _learning_rate)
                plot(loss_list, _predictions_series, batchY, _answers_series, batchLabels)
                
    if input("Do you want to save the model?(y/n): ") == "y":
        _save_path = saver.save(sess, save_path)
        print("Model saved in file: %s" % _save_path)


plt.ioff()
plt.show()
