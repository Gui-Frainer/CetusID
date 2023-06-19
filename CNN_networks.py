import os
from tensorflow.keras import backend
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPool2D, Conv2D
from tensorflow.keras.models import Sequential

#CNN for dolphin detection
def network_2D_CNN1(conv_layers, fc_layers, conv_filters, dropout_rate, 
               conv_kernel,max_pooling_size,fc_units,epochs,batch_size, image_size):
    
    model = Sequential()
    model.add(Conv2D(filters = conv_filters, kernel_size = conv_kernel, input_shape = (image_size,image_size,1)
                     ,activation = 'relu'))
    model.add(Dropout(rate = dropout_rate))
    model.add(MaxPool2D(pool_size = max_pooling_size))
    
    for i in range(conv_layers):
        model.add(Conv2D(filters = conv_filters, kernel_size = conv_kernel, activation = 'relu'))
        model.add(Dropout(rate=dropout_rate))
        model.add(MaxPool2D(pool_size=max_pooling_size))
        
    model.add(Flatten())
    for i in range(fc_layers):
        model.add(Dense(units = fc_units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))
        
    model.add(Dense(2, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    return model

def network_CNN1(image_size):

    conv_layers = 2
    fc_layers = 1
    max_pooling_size = 4
    dropout_rate = 0.4
    conv_filters = 32
    conv_kernel = 4
    fc_units = 64
    epochs = 50
    batch_size = 32
    model = network_2D_CNN1(conv_layers, fc_layers, conv_filters, dropout_rate, 
               conv_kernel,max_pooling_size,fc_units,epochs,batch_size, image_size)
    
    return model


#CNN for species identification

def network_2D_CNN2(conv_layers, fc_layers, conv_filters, dropout_rate, 
               conv_kernel,max_pooling_size,fc_units,epochs,batch_size,image_size,species_dir):
    
    model = Sequential()
    model.add(Conv2D(filters = conv_filters, kernel_size = conv_kernel, input_shape = (image_size,image_size,1)
                     ,activation = 'relu'))
    model.add(Dropout(rate = dropout_rate))
    model.add(MaxPool2D(pool_size = max_pooling_size))
    
    for i in range(conv_layers):
        model.add(Conv2D(filters = conv_filters, kernel_size = conv_kernel, activation = 'relu'))
        model.add(Dropout(rate=dropout_rate))
        model.add(MaxPool2D(pool_size=max_pooling_size))
        
    model.add(Flatten())
    for i in range(fc_layers):
        model.add(Dense(units = fc_units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))
        
    
    model.add(Dense(int(len(next(os.walk(species_dir))[1])), activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    return model

def network_CNN2(image_size, species_dir):

    conv_layers = 2
    fc_layers = 1
    max_pooling_size = 4
    dropout_rate = 0.4
    conv_filters = 32
    conv_kernel = 4
    fc_units = 64
    epochs = 50
    batch_size = 32
    model = network_2D_CNN2(conv_layers, fc_layers, conv_filters, dropout_rate, 
               conv_kernel,max_pooling_size,fc_units,epochs,batch_size,image_size, species_dir)
    
    return model



#Transfer learning
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.applications import ResNet101V2, ResNet152V2, ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def network_CNN1_TF(image_size):
        INPUT_SHAPE = (image_size,image_size,3)

        base_model = ResNet152V2(weights="imagenet",
            input_shape=INPUT_SHAPE,
            include_top=False)  


        
        base_model.trainable = False

        
        inputs = Input(shape=INPUT_SHAPE)

        
        
        
        x = base_model(inputs)

        x = Flatten()(x)
        outputs = Dense(2, activation='softmax')(x)
        model = Model(inputs, outputs)
        

        
        
        
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        return model
	
def network_CNN2_TF(image_size, species_dir):
        INPUT_SHAPE = (image_size,image_size,3)

        base_model = ResNet152V2(weights="imagenet",
            input_shape=INPUT_SHAPE,
            include_top=False)  


        
        base_model.trainable = False

        
        inputs = Input(shape=INPUT_SHAPE)

        
        
        
        x = base_model(inputs)

        x = Flatten()(x)
        outputs = Dense(int(len(next(os.walk(species_dir))[1])), activation='softmax')(x)
        model = Model(inputs, outputs)
        

        
        
        
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        return model