# Automatic-Speech-Recognition
### 使用的套件資源
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Activation, Conv1D, Lambda, Add, Multiply, BatchNormalization
    from tensorflow.keras.optimizers import Adam, SGD
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    %matplotlib inline
    import random
    import pickle
    import glob
    from tqdm import tqdm
    import os
    from python_speech_features import mfcc
    import scipy.io.wavfile as wav
    import librosa
    from IPython.display import Audio
    import csv
### 訓練檔案格式轉換
     with open('./train-toneless_update.csv', newline='', errors='ignore') as csvfile:
          rows = csv.reader(csvfile)
     for row in rows:
        if row[0] != 'id':
            a=row[0]
            b=row[1].lower()
            f=open('./train/txt/' + a + '.txt', 'w')
            f.write(b)
            f.close()
### 讀取音檔 
    def get_wav_files(wav_path):
    wav_files = []
    for (dirpath, dirname, filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                filename_path = os.path.join(dirpath, filename)
                wav_files.append(filename_path)
    return wav_files
### 讀取文字檔
    def get_tran_texts(wav_files, tran_path):
    tran_texts = []
    for wav_file in wav_files:
        basename = os.path.basename(wav_file)
        x = os.path.splitext(basename)[0]
        tran_file = os.path.join(tran_path, x+'.txt')
        if os.path.exists(tran_file) is False:
            return None
        fd = open(tran_file, 'r')
        text = fd.readline()
        tran_texts.append(text.split('\n')[0])
        fd.close()
    return tran_texts
 ### 處理音訊
      mfcc_dim = 13

      def load_and_trim(path):
          audio, sr = librosa.load(path)
          energy = librosa.feature.rms(audio)
          frames = np.nonzero(energy >= np.max(energy)/10)
          indices = librosa.core.frames_to_samples(frames)[1]
          audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

          return audio, sr
### 視覺化呈現資料
    def visualize(index):
    path = paths[index]
    text = texts[index]
    print('Audio Text:', text)
    
    audio, sr = load_and_trim(path)
    plt.figure(figsize=(12, 3))
    plt.plot(np.arange(len(audio)), audio)
    plt.title('Raw Audio Signal')
    plt.xlabel('Time')
    plt.ylabel('Audio Amplitude')
    plt.show()
    
    feature = mfcc(audio, sr, numcep=mfcc_dim, nfft=551)
    print('Shape of MFCC:', feature.shape)
    
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(feature, cmap=plt.cm.jet, aspect='auto')
    plt.title('Normalized MFCC')
    plt.ylabel('Time')
    plt.xlabel('MFCC Coefficient')
    plt.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
    ax.set_xticks(np.arange(0, 13, 2), minor=False);
    plt.show()
    
    return path
![image](https://github.com/t109318170KevinSu/Automatic-Speech-Recognition/blob/main/%E5%9C%961.png)
![image](https://github.com/t109318170KevinSu/Automatic-Speech-Recognition/blob/main/MFCC.png)
### 特徵處理
    samples = random.sample(features, 100)
    samples = np.vstack(samples)

    mfcc_mean = np.mean(samples, axis=0)
    mfcc_std = np.std(samples, axis=0)
    print(mfcc_mean)
    print(mfcc_std)

    features = [(feature - mfcc_mean) / (mfcc_std + 1e-14) for feature in features]
### 讀取文字&創立字典
    chars = {}
    for text in texts:
        text = text.lower()
        for c in text:
            chars[c] = chars.get(c, 0) + 1

    chars = sorted(chars.items(), key=lambda x: x[1], reverse=True)
    chars = [char[0] for char in chars]
    print(len(chars), chars[:100])

    char2id = {c: i for i, c in enumerate(chars)}
    id2char = {i: c for i, c in enumerate(chars)}
    print(char2id)
### 定義輸入&輸出
      data_index = np.arange(total)
      np.random.shuffle(data_index)
      train_size = int(0.9 * total)
      test_size = total - train_size
      train_index = data_index[:train_size]
      test_index = data_index[train_size:]

      X_train = [features[i] for i in train_index]
      Y_train = [texts[i] for i in train_index]
      X_test = [features[i] for i in test_index]
      Y_test = [texts[i] for i in test_index]

      batch_size = 4

      def batch_generator(x, y, batch_size=batch_size):  
          offset = 0
          while True:
              offset += batch_size

              if offset == batch_size or offset >= len(x):
                  data_index = np.arange(len(x))
                  np.random.shuffle(data_index)
                  x = [x[i] for i in data_index]
                  y = [y[i] for i in data_index]
                  offset = batch_size

              X_data = x[offset - batch_size: offset]
              Y_data = y[offset - batch_size: offset]

              X_maxlen = max([X_data[i].shape[0] for i in range(batch_size)])
              Y_maxlen = max([len(Y_data[i]) for i in range(batch_size)])

              X_batch = np.zeros([batch_size, X_maxlen, mfcc_dim])
              Y_batch = np.ones([batch_size, Y_maxlen]) * len(char2id)
              X_length = np.zeros([batch_size, 1], dtype='int32')
              Y_length = np.zeros([batch_size, 1], dtype='int32')
              for i in range(batch_size):
                  X_length[i, 0] = X_data[i].shape[0]
                  X_batch[i, :X_length[i, 0], :] = X_data[i]

                  Y_length[i, 0] = len(Y_data[i])
                  Y_batch[i, :Y_length[i, 0]] = [char2id[c] for c in Y_data[i]]

              inputs = {'X': X_batch, 'Y': Y_batch, 'X_length': X_length, 'Y_length': Y_length}
              outputs = {'ctc': np.zeros([batch_size])}
              yield (inputs, outputs)
### 定義模型架構
      epochs = 30
      num_blocks = 3
      filters = 128

      X = Input(shape=(None, mfcc_dim,), dtype='float32', name='X')
      Y = Input(shape=(None,), dtype='float32', name='Y')
      X_length = Input(shape=(1,), dtype='int32', name='X_length')
      Y_length = Input(shape=(1,), dtype='int32', name='Y_length')

      def conv1d(inputs, filters, kernel_size, dilation_rate):
          return Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='causal', activation=None, dilation_rate=dilation_rate)(inputs)

      def batchnorm(inputs):
          return BatchNormalization()(inputs)

      def activation(inputs, activation):
          return Activation(activation)(inputs)

      def res_block(inputs, filters, kernel_size, dilation_rate):
          hf = activation(batchnorm(conv1d(inputs, filters, kernel_size, dilation_rate)), 'tanh')
          hg = activation(batchnorm(conv1d(inputs, filters, kernel_size, dilation_rate)), 'sigmoid')
          h0 = Multiply()([hf, hg])

          ha = activation(batchnorm(conv1d(h0, filters, 1, 1)), 'tanh')
          hs = activation(batchnorm(conv1d(h0, filters, 1, 1)), 'tanh')

          return Add()([ha, inputs]), hs

      h0 = activation(batchnorm(conv1d(X, filters, 1, 1)), 'tanh')
      shortcut = []
      for i in range(num_blocks):
          for r in [1, 2, 4, 8, 16]:
              h0, s = res_block(h0, filters, 7, r)
              shortcut.append(s)

      h1 = activation(Add()(shortcut), 'relu')
      h1 = activation(batchnorm(conv1d(h1, filters, 1, 1)), 'relu')
      Y_pred = activation(batchnorm(conv1d(h1, len(char2id) + 1, 1, 1)), 'softmax')
      sub_model = Model(inputs=X, outputs=Y_pred)

      def calc_ctc_loss(args):
          y, yp, ypl, yl = args
          return K.ctc_batch_cost(y, yp, ypl, yl)

      ctc_loss = Lambda(calc_ctc_loss, output_shape=(1,), name='ctc')([Y, Y_pred, X_length, Y_length])
      model = Model(inputs=[X, Y, X_length, Y_length], outputs=ctc_loss)
      optimizer =SGD(lr=0.002, momentum=0.9, nesterov=True, clipnorm=5)
      model.compile(loss={'ctc': lambda ctc_true, ctc_pred: ctc_pred}, optimizer=optimizer)

      checkpointer = ModelCheckpoint(filepath='asr.h5', verbose=0)
      lr_decay = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=1, min_lr=0.000)

      history = model.fit_generator(
          generator=batch_generator(X_train, Y_train), 
          steps_per_epoch=len(X_train) // batch_size,
          epochs=epochs, 
          validation_data=batch_generator(X_test, Y_test), 
          validation_steps=len(X_test) // batch_size, 
          callbacks=[checkpointer, lr_decay])
 ### 訓練結果
  
