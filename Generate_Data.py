import numpy as np

from os import path
import os, random

import osureader

import subprocess
from scipy.io.wavfile import read


sampling_rate = 8000

##    folder = path.dirname(__file__)
##    data_folder = path.join(folder, "Training Songs")
data_folder = "C:\\Users\\Olivier\\AppData\\Local\\osu!\\Songs"


def validateBeatmap(beatmap):
    foundanybpm = False
    for t in beatmap.TimingPoints:
        if t[6] == 1:
            foundanybpm = True
            if 60000 / t[1] <= 20 or 60000 / t[1] >= 500:
                return False
    if not foundanybpm:
        return False
    return True

def getRandomSong():
    #get random beatmap dir
    notFoundMap = True
    while notFoundMap:
        beatmap_folder = path.join(data_folder, random.choice(os.listdir(data_folder)))
        beatmap_list = [f for f in os.listdir(beatmap_folder) if f[-4:] == ".osu"]
        if len(beatmap_list) == 0:
            pass
            #print("Empty beatmap folder detected. Deleting the folder from training directory")
            #os.remove(beatmap_folder)
        else:
            beatmap_path = path.join(beatmap_folder, random.choice(beatmap_list))
    
            #open the beatmap
            beatmap = osureader.readBeatmap(beatmap_path)
            #find the audio
            audio_path = path.join(beatmap_folder, beatmap.AudioFilename)
            if validateBeatmap(beatmap) and path.isfile(audio_path) :
                notFoundMap = False
            
    #find the audio
    audio_path = path.join(beatmap_folder, beatmap.AudioFilename)
    
    #open the audio
    wav_path = path.join(beatmap_folder, "audio.wav.wav")
    if not path.exists(wav_path):
        subprocess.call(['ffmpeg', '-i', audio_path, "-ar", str(sampling_rate),
                         "-ac", "1",
                       wav_path])
    audio = read(wav_path)
    
    if not audio[0] == sampling_rate:
        os.remove(wav_path)
        subprocess.call(['ffmpeg', '-i', audio_path, "-ar", str(sampling_rate),
                         "-ac", "1",
                       wav_path])
        audio = read(wav_path)
        
    audio = audio[1]

    return (audio, beatmap)

def getBPMatTime(beatmap, time):
    bpm = 0
    index = 0
    while beatmap.TimingPoints[index][0] <= time or bpm == 0:
        if beatmap.TimingPoints[index][6] == 1:
            if 60000 / beatmap.TimingPoints[index][1] > 20 and 60000 / beatmap.TimingPoints[index][1] < 500:
                bpm = 60000 / beatmap.TimingPoints[index][1]
        index += 1
        if index == len(beatmap.TimingPoints):
            break
    return bpm

def generateData(num_batches, audio_size):
    x_list = []
    y_list = []

    batches_made = 0
    
    time = 10000
    audio, beatmap = getRandomSong()
    song_length = len(audio) / sampling_rate * 1000
    end_time = song_length - 10000 - (audio_size / sampling_rate * 1000)

    while batches_made < num_batches:
        if batches_made % 100 == 0:
            print(batches_made)
        if time > end_time:
            time = 10000
            audio, beatmap = getRandomSong()
            song_length = len(audio) / sampling_rate * 1000
            end_time = song_length - 10000 - (audio_size / sampling_rate * 1000)

        audio_index = int(time * sampling_rate / 1000)
        x_list.append(audio[audio_index:audio_index+audio_size])
        y_list.append(getBPMatTime(beatmap, time))

        batches_made += 1

        time += random.randint(audio_size / sampling_rate * 1000, audio_size / sampling_rate * 2000)
    
    
    x = np.hstack(x_list)
    y = np.hstack(y_list)

    x = np.divide(x, 32767)

    x = x.reshape((num_batches, audio_size))
    y = y.reshape((num_batches, 1))
    
    return (x.astype(np.float32, copy=False), y.astype(np.float32, copy=False))

def generateTrainingData(num_batches):
    folder = path.dirname(__file__)
    data_folder = path.join(folder, "Training_data")
    audio_path = path.join(data_folder, "train-audio.npy")
    labels_path = path.join(data_folder, "train-labels.npy")
    x, y = generateData(num_batches, 40000)
    try:
        x_old = np.load(audio_path)
        y_old = np.load(labels_path)
        print("Loaded existing training data")
        x = np.concatenate((x_old, x), axis=0)
        y = np.concatenate((y_old, y), axis=0)
    except:
        pass
    np.save(audio_path, x)
    np.save(labels_path, y)

def loadTrainingData():
    folder = path.dirname(__file__)
    data_folder = path.join(folder, "Training_data")
    audio_path = path.join(data_folder, "train-audio.npy")
    labels_path = path.join(data_folder, "train-labels.npy")
    x = np.load(audio_path)
    y = np.load(labels_path)
    return (x, y)
