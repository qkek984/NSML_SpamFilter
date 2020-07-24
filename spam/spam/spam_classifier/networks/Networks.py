from keras import Input, Model
#from keras.applications.resnet_v2 import ResNet50V2
#from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Flatten, Dense, Dropout
from keras.layers import GlobalAveragePooling2D
from os import path


#def frozen_resnet(input_size, n_classes, local_weights="/resnets/resnet50v2_notop.h5"):
def frozen_networks(input_size, n_classes, local_weights='imagenet'):
    if local_weights and path.exists(local_weights):
        print(f'Using {local_weights} as local weights.')
        model_ = InceptionResNetV2(
            include_top=False,#fine tuning을 위한 false
            input_tensor=Input(shape=input_size),
            weights=local_weights)
    else:
        print(
            f'Could not find local weights {local_weights} for Model. Using remote weights.')
        model_ = InceptionResNetV2(
            include_top=False,
            input_tensor=Input(shape=input_size))

    for layer in model_.layers:
        layer.trainable = False#불러온 모델의 웨이트를 학습하지 않도록 설정
    #x = Flatten(input_shape=model_.output_shape[1:])(model_.layers[-1].output)
    x = GlobalAveragePooling2D()(model_.layers[-1].output)
    #x = Dense(512, activation='relu')(x)
    x = Dense(n_classes, activation='softmax')(x)
    frozen_model = Model(model_.input, x)

    frozen_model.summary()#모델구조 출력
    return frozen_model