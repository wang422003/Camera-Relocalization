from scipy.misc import imread, imresize

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD
from googlenet_custom_layers import PoolHelper,LRN


def create_posenet(weights_path=None):
    # creates PoseNet base on the GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)

    # Set the input dimension (the same as GoogLeNet)
    input = Input(shape=(3, 224, 224))

    # Convolutional layer
    conv1_7x7_s2 = Convolution2D(64,7,7,subsample=(2,2),border_mode='same',activation='relu',name='conv1/7x7_s2',
                                 W_regularizer=l2(0.0002))(input)

    # Zero padding
    conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)
    
    pool1_helper = PoolHelper()(conv1_zero_pad)

    # Max Pooling layer
    pool1_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool1/3x3_s2')(pool1_helper)

    # Normalizatio after max pooling
    pool1_norm1 = LRN(name='pool1/norm1')(pool1_3x3_s2)

    # Reduction layer
    conv2_3x3_reduce = Convolution2D(64,1,1,border_mode='same',activation='relu',name='conv2/3x3_reduce',
                                     W_regularizer=l2(0.0002))(pool1_norm1)

    # Convolutional layer
    conv2_3x3 = Convolution2D(192,3,3,border_mode='same',activation='relu',name='conv2/3x3',
                              W_regularizer=l2(0.0002))(conv2_3x3_reduce)

    # Normalizaion
    conv2_norm2 = LRN(name='conv2/norm2')(conv2_3x3)

    # Zero padding
    conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_norm2)
    
    pool2_helper = PoolHelper()(conv2_zero_pad)

    # Max Pooling layer
    pool2_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool2/3x3_s2')(pool2_helper)
    
    ## First inception module
    inception_3a_1x1 = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_3a/1x1',
                                     W_regularizer=l2(0.0002))(pool2_3x3_s2)
    
    inception_3a_3x3_reduce = Convolution2D(96,1,1,border_mode='same',activation='relu',name='inception_3a/3x3_reduce',
                                            W_regularizer=l2(0.0002))(pool2_3x3_s2)
    
    inception_3a_3x3 = Convolution2D(128,3,3,border_mode='same',activation='relu',name='inception_3a/3x3',
                                     W_regularizer=l2(0.0002))(inception_3a_3x3_reduce)
    
    inception_3a_5x5_reduce = Convolution2D(16,1,1,border_mode='same',activation='relu',name='inception_3a/5x5_reduce',
                                            W_regularizer=l2(0.0002))(pool2_3x3_s2)
    
    inception_3a_5x5 = Convolution2D(32,5,5,border_mode='same',activation='relu',name='inception_3a/5x5',
                                     W_regularizer=l2(0.0002))(inception_3a_5x5_reduce)
    
    inception_3a_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',
                                     name='inception_3a/pool')(pool2_3x3_s2)
    
    inception_3a_pool_proj = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_3a/pool_proj',
                                           W_regularizer=l2(0.0002))(inception_3a_pool)
    
    inception_3a_output = merge([inception_3a_1x1,inception_3a_3x3,inception_3a_5x5,inception_3a_pool_proj],
                                mode='concat',concat_axis=1,name='inception_3a/output')
    
    ## Second inception module

    # 1x1 convolution layer
    inception_3b_1x1 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_3b/1x1',
                                     W_regularizer=l2(0.0002))(inception_3a_output)

    # 3x3 reduction layerc and its onvulutional layer
    inception_3b_3x3_reduce = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_3b/3x3_reduce',
                                            W_regularizer=l2(0.0002))(inception_3a_output)
    
    inception_3b_3x3 = Convolution2D(192,3,3,border_mode='same',activation='relu',name='inception_3b/3x3',
                                     W_regularizer=l2(0.0002))(inception_3b_3x3_reduce)

    # 5x5 reduction layerc and its onvulutional layer
    inception_3b_5x5_reduce = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_3b/5x5_reduce',
                                            W_regularizer=l2(0.0002))(inception_3a_output)
    
    inception_3b_5x5 = Convolution2D(96,5,5,border_mode='same',activation='relu',name='inception_3b/5x5',
                                     W_regularizer=l2(0.0002))(inception_3b_5x5_reduce)
    # Max Pooling
    inception_3b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_3b/pool')(inception_3a_output)
    
    inception_3b_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_3b/pool_proj',
                                           W_regularizer=l2(0.0002))(inception_3b_pool)
    
    inception_3b_output = merge([inception_3b_1x1,inception_3b_3x3,inception_3b_5x5,inception_3b_pool_proj],
                                mode='concat',concat_axis=1,name='inception_3b/output')
    
    # Zero padding
    inception_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_output)
    
    pool3_helper = PoolHelper()(inception_3b_output_zero_pad)

    # Max Pooling
    pool3_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool3/3x3_s2')(pool3_helper)
    
    ## Third inception module
    inception_4a_1x1 = Convolution2D(192,1,1,border_mode='same',activation='relu',name='inception_4a/1x1',
                                     W_regularizer=l2(0.0002))(pool3_3x3_s2)
    
    inception_4a_3x3_reduce = Convolution2D(96,1,1,border_mode='same',activation='relu',name='inception_4a/3x3_reduce',
                                            W_regularizer=l2(0.0002))(pool3_3x3_s2)
    
    inception_4a_3x3 = Convolution2D(208,3,3,border_mode='same',activation='relu',name='inception_4a/3x3',
                                     W_regularizer=l2(0.0002))(inception_4a_3x3_reduce)
    
    inception_4a_5x5_reduce = Convolution2D(16,1,1,border_mode='same',activation='relu',name='inception_4a/5x5_reduce',
                                            W_regularizer=l2(0.0002))(pool3_3x3_s2)
    
    inception_4a_5x5 = Convolution2D(48,5,5,border_mode='same',activation='relu',name='inception_4a/5x5',
                                     W_regularizer=l2(0.0002))(inception_4a_5x5_reduce)
    
    inception_4a_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4a/pool')(pool3_3x3_s2)
    
    inception_4a_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_4a/pool_proj',
                                           W_regularizer=l2(0.0002))(inception_4a_pool)
    
    inception_4a_output = merge([inception_4a_1x1,inception_4a_3x3,inception_4a_5x5,inception_4a_pool_proj],
                                mode='concat',concat_axis=1,name='inception_4a/output')
    
    ## First classification branch

    # Average Pooling
    loss1_ave_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),name='loss1/ave_pool')(inception_4a_output)

    # Convolutional Layer for loss
    loss1_conv = Convolution2D(128,1,1,border_mode='same',activation='relu',name='loss1/conv',
                               W_regularizer=l2(0.0002))(loss1_ave_pool)
    # Flatten for the next full connection layer
    loss1_flat = Flatten()(loss1_conv)

    # Full connection layer
    loss1_fc = Dense(1024,activation='relu',name='loss1/fc',W_regularizer=l2(0.0002))(loss1_flat)

    # Dropout
    loss1_drop_fc = Dropout(0.7)(loss1_fc)

    # 3-D Position Regression
    cls1_fc_pose_xyz = Dense(3, name='posexyz1',W_regularizer=l2(0.0002))(loss1_drop_fc) # New and different from the original GoogLeNet. The classification is replaced by regression.

    # 4-D Orientation Regression
    cls1_fc_pose_wpqr = Dense(4,name='posewpqr1',W_regularizer=l2(0.0002))(loss1_drop_fc) # New

    ## Fourth inception module
    inception_4b_1x1 = Convolution2D(160,1,1,border_mode='same',activation='relu',name='inception_4b/1x1',
                                     W_regularizer=l2(0.0002))(inception_4a_output)
    
    inception_4b_3x3_reduce = Convolution2D(112,1,1,border_mode='same',activation='relu',name='inception_4b/3x3_reduce',
                                            W_regularizer=l2(0.0002))(inception_4a_output)
    
    inception_4b_3x3 = Convolution2D(224,3,3,border_mode='same',activation='relu',name='inception_4b/3x3',
                                     W_regularizer=l2(0.0002))(inception_4b_3x3_reduce)
    
    inception_4b_5x5_reduce = Convolution2D(24,1,1,border_mode='same',activation='relu',name='inception_4b/5x5_reduce',
                                            W_regularizer=l2(0.0002))(inception_4a_output)
    
    inception_4b_5x5 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='inception_4b/5x5',
                                     W_regularizer=l2(0.0002))(inception_4b_5x5_reduce)
    
    inception_4b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4b/pool')(inception_4a_output)
    
    inception_4b_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_4b/pool_proj',
                                           W_regularizer=l2(0.0002))(inception_4b_pool)
    
    inception_4b_output = merge([inception_4b_1x1,inception_4b_3x3,inception_4b_5x5,inception_4b_pool_proj],
                                mode='concat',concat_axis=1,name='inception_4b_output')
    
    ## Fifth inception module
    inception_4c_1x1 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_4c/1x1',
                                     W_regularizer=l2(0.0002))(inception_4b_output)
    
    inception_4c_3x3_reduce = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_4c/3x3_reduce',
                                            W_regularizer=l2(0.0002))(inception_4b_output)
    
    inception_4c_3x3 = Convolution2D(256,3,3,border_mode='same',activation='relu',name='inception_4c/3x3',
                                     W_regularizer=l2(0.0002))(inception_4c_3x3_reduce)
    
    inception_4c_5x5_reduce = Convolution2D(24,1,1,border_mode='same',activation='relu',name='inception_4c/5x5_reduce',
                                            W_regularizer=l2(0.0002))(inception_4b_output)
    
    inception_4c_5x5 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='inception_4c/5x5',
                                     W_regularizer=l2(0.0002))(inception_4c_5x5_reduce)
    
    inception_4c_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4c/pool')(inception_4b_output)
    
    inception_4c_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_4c/pool_proj',
                                           W_regularizer=l2(0.0002))(inception_4c_pool)
    
    inception_4c_output = merge([inception_4c_1x1,inception_4c_3x3,inception_4c_5x5,inception_4c_pool_proj],
                                mode='concat',concat_axis=1,name='inception_4c/output')
    
    ## Sixth inception module
    inception_4d_1x1 = Convolution2D(112,1,1,border_mode='same',activation='relu',name='inception_4d/1x1',
                                     W_regularizer=l2(0.0002))(inception_4c_output)
    
    inception_4d_3x3_reduce = Convolution2D(144,1,1,border_mode='same',activation='relu',name='inception_4d/3x3_reduce',
                                            W_regularizer=l2(0.0002))(inception_4c_output)
    
    inception_4d_3x3 = Convolution2D(288,3,3,border_mode='same',activation='relu',name='inception_4d/3x3',
                                     W_regularizer=l2(0.0002))(inception_4d_3x3_reduce)
    
    inception_4d_5x5_reduce = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_4d/5x5_reduce',
                                            W_regularizer=l2(0.0002))(inception_4c_output)
    
    inception_4d_5x5 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='inception_4d/5x5',
                                     W_regularizer=l2(0.0002))(inception_4d_5x5_reduce)
    
    inception_4d_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4d/pool')(inception_4c_output)
    
    inception_4d_pool_proj = Convolution2D(64,1,1,border_mode='same',activation='relu',name='inception_4d/pool_proj',
                                           W_regularizer=l2(0.0002))(inception_4d_pool)
    
    inception_4d_output = merge([inception_4d_1x1,inception_4d_3x3,inception_4d_5x5,inception_4d_pool_proj],
                                mode='concat',concat_axis=1,name='inception_4d/output')
    
    ## Second classification branch

    # Average Pooling
    loss2_ave_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),name='loss2/ave_pool')(inception_4d_output)

    # Convolutional Layer for loss
    loss2_conv = Convolution2D(128,1,1,border_mode='same',activation='relu',name='loss2/conv',
                               W_regularizer=l2(0.0002))(loss2_ave_pool)
    # Flatten for the next full connection layer
    loss2_flat = Flatten()(loss2_conv)

    # Fully connected layer
    loss2_fc = Dense(1024,activation='relu',name='loss2/fc',W_regularizer=l2(0.0002))(loss2_flat)

    # Dropout layer
    loss2_drop_fc = Dropout(0.7)(loss2_fc)

    # 3-D position regression
    cls2_fc_pose_xyz = Dense(3, name='posexyz2',W_regularizer=l2(0.0002))(loss2_drop_fc) # New

    # 4-D orientation regression
    cls2_fc_pose_wpqr = Dense(4,name='posewpqr2',W_regularizer=l2(0.0002))(loss2_drop_fc) # New


    ## Seventh inceptio module
    inception_4e_1x1 = Convolution2D(256,1,1,border_mode='same',activation='relu',name='inception_4e/1x1',
                                     W_regularizer=l2(0.0002))(inception_4d_output)
    
    inception_4e_3x3_reduce = Convolution2D(160,1,1,border_mode='same',activation='relu',name='inception_4e/3x3_reduce',
                                            W_regularizer=l2(0.0002))(inception_4d_output)
    
    inception_4e_3x3 = Convolution2D(320,3,3,border_mode='same',activation='relu',name='inception_4e/3x3',
                                     W_regularizer=l2(0.0002))(inception_4e_3x3_reduce)
    
    inception_4e_5x5_reduce = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_4e/5x5_reduce',
                                            W_regularizer=l2(0.0002))(inception_4d_output)
    
    inception_4e_5x5 = Convolution2D(128,5,5,border_mode='same',activation='relu',name='inception_4e/5x5',
                                     W_regularizer=l2(0.0002))(inception_4e_5x5_reduce)
    
    inception_4e_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_4e/pool')(inception_4d_output)
    
    inception_4e_pool_proj = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_4e/pool_proj',
                                           W_regularizer=l2(0.0002))(inception_4e_pool)
    
    inception_4e_output = merge([inception_4e_1x1,inception_4e_3x3,inception_4e_5x5,inception_4e_pool_proj],
                                mode='concat',concat_axis=1,name='inception_4e/output')
    
    # Zero padding
    inception_4e_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_output)
    
    pool4_helper = PoolHelper()(inception_4e_output_zero_pad)

    # Max pooling
    pool4_3x3_s2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool4/3x3_s2')(pool4_helper)
    
    ## Eighth inception module
    inception_5a_1x1 = Convolution2D(256,1,1,border_mode='same',activation='relu',name='inception_5a/1x1',
                                     W_regularizer=l2(0.0002))(pool4_3x3_s2)
    
    inception_5a_3x3_reduce = Convolution2D(160,1,1,border_mode='same',activation='relu',name='inception_5a/3x3_reduce',
                                            W_regularizer=l2(0.0002))(pool4_3x3_s2)
    
    inception_5a_3x3 = Convolution2D(320,3,3,border_mode='same',activation='relu',name='inception_5a/3x3',
                                     W_regularizer=l2(0.0002))(inception_5a_3x3_reduce)
    
    inception_5a_5x5_reduce = Convolution2D(32,1,1,border_mode='same',activation='relu',name='inception_5a/5x5_reduce',
                                            W_regularizer=l2(0.0002))(pool4_3x3_s2)
    
    inception_5a_5x5 = Convolution2D(128,5,5,border_mode='same',activation='relu',name='inception_5a/5x5',
                                     W_regularizer=l2(0.0002))(inception_5a_5x5_reduce)
    
    inception_5a_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_5a/pool')(pool4_3x3_s2)
    
    inception_5a_pool_proj = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_5a/pool_proj',
                                           W_regularizer=l2(0.0002))(inception_5a_pool)
    
    inception_5a_output = merge([inception_5a_1x1,inception_5a_3x3,inception_5a_5x5,inception_5a_pool_proj],
                                mode='concat',concat_axis=1,name='inception_5a/output')
    
    ## Ninth inception module
    inception_5b_1x1 = Convolution2D(384,1,1,border_mode='same',activation='relu',name='inception_5b/1x1',
                                     W_regularizer=l2(0.0002))(inception_5a_output)
    
    inception_5b_3x3_reduce = Convolution2D(192,1,1,border_mode='same',activation='relu',name='inception_5b/3x3_reduce',
                                            W_regularizer=l2(0.0002))(inception_5a_output)
    
    inception_5b_3x3 = Convolution2D(384,3,3,border_mode='same',activation='relu',name='inception_5b/3x3',
                                     W_regularizer=l2(0.0002))(inception_5b_3x3_reduce)
    
    inception_5b_5x5_reduce = Convolution2D(48,1,1,border_mode='same',activation='relu',name='inception_5b/5x5_reduce',
                                            W_regularizer=l2(0.0002))(inception_5a_output)
    
    inception_5b_5x5 = Convolution2D(128,5,5,border_mode='same',activation='relu',name='inception_5b/5x5',
                                     W_regularizer=l2(0.0002))(inception_5b_5x5_reduce)
    
    inception_5b_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_5b/pool')(inception_5a_output)
    
    inception_5b_pool_proj = Convolution2D(128,1,1,border_mode='same',activation='relu',name='inception_5b/pool_proj',
                                           W_regularizer=l2(0.0002))(inception_5b_pool)
    
    inception_5b_output = merge([inception_5b_1x1,inception_5b_3x3,inception_5b_5x5,inception_5b_pool_proj],
                                mode='concat',concat_axis=1,name='inception_5b/output')
    
    ## the following layers only for bayesian_posenet

    # Dropout layer(Bayesian)
    pool5_drop_bay = Dropout(0.5)(inception_5b_output)  #bay

    # Average pooling
    pool5_7x7_s1 = AveragePooling2D(pool_size=(7,7),strides=(1,1),name='pool5/7x7_s2')(pool5_drop_bay)

    # Flatten
    loss3_flat = Flatten()(pool5_7x7_s1)

    # Fully connected layer
    loss3_poseregressor = Dense(2048,,activation='relu', name='loss3/poseregressor',W_regularizer=l2(0.0002))(loss3_flat)

    # Dropout
    pool5_drop_7x7_s1 = Dropout(0.5)(loss3_poseregressor)

    # 3-D position regression
    cls3_fc_pose_xyz = Dense(3, name='posexyz3',W_regularizer=l2(0.0002))(pool5_drop_7x7_s1) # New

    # 4-D orientation regression
    cls3_fc_pose_wpqr = Dense(4,name='posewpqr3',W_regularizer=l2(0.0002))(pool5_drop_7x7_s1) # New    

    posenet = Model(input=input, output=[cls1_fc_pose_xyz,cls2_fc_pose_xyz,cls3_fc_pose_xyz,
                                         cls1_fc_pose_wpqr,cls2_fc_pose_wpqr,cls3_fc_pose_wpqr])
    
    if weights_path:
        posenet.load_weights(weights_path)

    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #posenet.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return posenet



if __name__ == "__main__":

    ## Section 1 : test the pretrained model with the single picture
    img = imresize(imread('cat.jpg', mode='RGB'), (224, 224)).astype(np.float32)
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    # Test pretrained model
    model = create_posenet('posenet_weights.h5')  # The model must be there
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mean_squard_error')
    out = model.predict(img) # note: the model has three sets of outputs

    print np.argmax(out[2])
 ##########################################################3333
    # Section 2 : Example to fine-tune on 3000 samples from Cifar10

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = 10
    batch_size = 16
    nb_epoch = 10

    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)

    # Load our model
    model = create_posenet()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mean_squard_error')

    # Start Fine-tuning.
    # Notice that PoseNet takes 3 sets of [position, orientation] for outputs, one for each auxillary regressor
    model.fit(X_train, [Y_train, Y_train, Y_train],
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, [Y_valid, Y_valid, Y_valid]),
              )

    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    # Combine 3 set of outputs using averaging
    predictions_valid = sum(predictions_valid)/len(predictions_valid)

    # Cross-entropy loss score
    score = log_loss(Y_valid, predictions_valid)