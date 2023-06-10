from __future__ import division


from model import TVdist,DINet
import tensorflow as tf
import h5py, yaml
import math
from argparse import ArgumentParser
import cv2,tensorflow.keras
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import sys,os
import numpy as np
from utilities import preprocess_images2, preprocess_maps2,postprocess_predictions,postprocess_predictions3
import time

def scheduler(epoch):
    if epoch%2==0 and epoch!=0:
        lr = K.get_value(model.optimizer.lr)
        if lr > 1e-7:
          K.set_value(model.optimizer.lr, lr*.1)
          print("lr changed to {}".format(lr*.1))
    return K.get_value(model.optimizer.lr)

def lr_scheduler(epoch, mode='progressive_drops'):
        '''if lr_dict.has_key(epoch):
            lr = lr_dict[epoch]
            print 'lr: %f' % lr'''
        lr_base = K.get_value(model.optimizer.lr)
        lr_power = 0.9
        epochs =10
        if mode is 'power_decay':
            # original lr scheduler
            lr = lr_base * ((1 - float(epoch)/epochs) ** lr_power)
        if mode is 'exp_decay':
            # exponential decay
            lr = (float(lr_base) ** float(lr_power)) ** float(epoch+1)
        # adam default lr
        if mode is 'adam':
            lr = 0.001

        if mode is 'progressive_drops':
            # drops as progression proceeds, good for sgd
            if epoch > 0.5 * epochs:
                lr = 1e-7
            elif epoch > 0.3 * epochs:
                lr = 1e-6
            elif epoch > 0.1 * epochs:
                lr = 1e-5
            else:
                lr = 1e-4
        K.set_value(model.optimizer.lr, lr)
        print('lr: %f' % lr)
        return K.get_value(model.optimizer.lr)

## r,c  = h,w

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,images,maps,conf, batch_size=32, shuffle=True,mirror=False):
        'Initialization'
        self.batch_size = batch_size
        self.conf = conf
        self.images =images
        self.maps =maps
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.mirror = mirror
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_IDs_temp.sort()
        # Generate data
        X,Y  = self.__data_generation(list_IDs_temp)

        return [X], [Y]


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        # print ('shuffling!')
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)
        shape_r =  self.conf['shape_r']  
        shape_c =  self.conf['shape_c']    

        index = list_IDs_temp
        # print index
        imagesa= [self.images[idx] for idx in index]
        # print imagesa
        mapsa = [self.maps[idx] for idx in index]    

        X= preprocess_images2(imagesa[0:len(index)], shape_r, shape_c,mirror=self.mirror)
        Y= preprocess_maps2(mapsa[0:len(index)], shape_r, shape_c,mirror=self.mirror)


        return X,Y
        
    
def generator_test(b_s, imgs_test_path):
    images = [imgs_test_path + f for f in os.listdir(imgs_test_path) if f.endswith(('.jpg', '.jpeg', '.png' , '.tif'))]
    images.sort()
#    print b_s
#    print len(images)
    counter = 0
    while True:
        yield preprocess_images2(images[counter:counter + b_s], shape_r, shape_c)
        counter = (counter + b_s) % len(images)


if __name__ == '__main__':
    parser = ArgumentParser(description='DINet')
    parser.add_argument('--batch_size', type=int,
                        help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 0.0001)')
    # parser.add_argument('--weight_decay', type=float, default=0.0,
    #                     help='weight decay (default: 0.0)')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='config file path (default: config.yaml)')
    parser.add_argument('--exp_id', default='0', type=str,
                        help='exp id (default: 0)')
    parser.add_argument('--database', default='Salicon', type=str,
                        help='database name (default: Salicon)')
    parser.add_argument('--phase', default='test', type=str,
                        help='phase (default: train')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pretrained weight (default: False')       
    parser.add_argument('--trainbn', default= False,
                        help='train bn layer? (default: False')               

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
    print('phase: ' + args.phase)
    print('exp id: ' + args.exp_id)
    print('database: ' + args.database)
    print('lr: ' + str(args.lr))
    # print('model: ' + args.model)
    print('batch_size: ' + str(args.batch_size))
    config.update(config[args.database])
    # config.update(config[args.model])


    b_s= args.batch_size 
    shape_r =  config['shape_r']  
    shape_c =  config['shape_c'] 
    imgs_train_path = config['imgs_train_path'] 
    maps_train_path = config['maps_train_path'] 
    imgs_val_path = config['imgs_val_path'] 
    maps_val_path = config['maps_val_path'] 

    model =DINet(img_cols=shape_c, img_rows=shape_r,train_bn= args.trainbn) 
    model.compile(Adam(lr=args.lr), loss= [TVdist]) 
    # model.compile(Adam(lr=1e-4), loss= [TVdist,TVdist],loss_weights=[0.5,0.5]) 
    model.summary()
    train_images = [imgs_train_path + f for f in os.listdir(imgs_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    train_maps = [maps_train_path + f for f in os.listdir(maps_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    val_images = [imgs_val_path + f for f in os.listdir(imgs_val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    val_maps = [maps_val_path + f for f in os.listdir(maps_val_path) if f.endswith(('.jpg', '.jpeg', '.png'))]    
    train_images.sort()
    train_maps.sort()
    val_images.sort()
    val_maps.sort()
    # print val_images
    if args.phase == 'train':    
        if args.pretrained == True:
            print("Load weights DINet")
            weight_file = 'models/'+ 'DINet.TVdist.EXP0-Salicon-lr=0.0001-bs=10.05-0.2638-0.2038.pkl'
            model.load_weights(weight_file)  
            print (weight_file)
        lr_decay = LearningRateScheduler(scheduler)

        nb_imgs_train =  len(train_images) 
        nb_imgs_val =  len(val_images) 
        train_index = range(nb_imgs_train)
        val_index = range(nb_imgs_val)
        print (nb_imgs_train,nb_imgs_val)
        print("Training DINet")

        checkpointdir= 'models/DINet.TVdist.'+ 'EXP{}-{}-lr={}-bs={}'.format(args.exp_id, args.database,str(args.lr),str(args.batch_size))
        print (checkpointdir)
        train_generator = DataGenerator(train_index,train_images,train_maps,config, b_s, shuffle=True, mirror= False)    
        val_generator = DataGenerator(val_index,val_images,val_maps,config, 1, shuffle=False, mirror= False)
        lr_decay = LearningRateScheduler(scheduler)
        model.fit_generator(generator=train_generator,epochs=args.epochs,steps_per_epoch= int(nb_imgs_train // b_s),
                            validation_data=val_generator, validation_steps= int(nb_imgs_val // 1),
                            callbacks=[EarlyStopping(patience=10),
                                       ModelCheckpoint(checkpointdir+'.{epoch:02d}-{val_loss:.4f}-{loss:.4f}.pkl', save_best_only=True),
                                        lr_decay]) 
    elif args.phase == "test":
        # path of output folder
        arg = 'DINet.TVdist.EXP0-Salicon-lr=0.0001-bs=10.05-0.2638-0.2038'
        print("Load weights DINet")

        weightfile='' #change loaded weights
        model.load_weights(weightfile)

        imgs_test_path = '/'
        output_folder = ''
        output_folder2 = ''

       
        if os.path.isdir(output_folder) is False:
            os.makedirs(output_folder)
        if os.path.isdir(output_folder2) is False:
            os.makedirs(output_folder2)      
        print("Predict saliency maps for " + imgs_test_path+ " at "+  output_folder)        

        file_names = [f for f in os.listdir(imgs_test_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        file_names.sort()

        nb_imgs_test = len(file_names)
        print (nb_imgs_test)
        start_time = time.time()    
        predictions0 = model.predict_generator(generator_test(b_s=1, imgs_test_path=imgs_test_path), nb_imgs_test)

        print (len(predictions0))
        elapsed_time2 = time.time() - start_time            
        print ("total model testing time: " , elapsed_time2)
             
        for pred, name in zip(predictions0, file_names):
            original_image = cv2.imread(imgs_test_path + name, 0)
            name = name[:-4] + '.jpg'
            res = postprocess_predictions(pred, original_image.shape[0], original_image.shape[1])
            res2 = postprocess_predictions3(pred, original_image.shape[0], original_image.shape[1],sigma=7)   #for getting better visualization
            cv2.imwrite(output_folder + '%s' % name, res.astype(int))
            cv2.imwrite(output_folder2 + '%s' % name, res2.astype(int))
        elapsed_time = time.time() - start_time            
        print ("total time: " , elapsed_time)
    else:
        raise NotImplementedError