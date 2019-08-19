from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense
from keras.models import Model

def vgg2c():
	model = VGG16(weights='imagenet', include_top=False)

	#Create your own input format (here 3x200x200)
	input = Input(shape=(224,224,3),name = 'image_input')

	#Use the generated model 
	output_vgg16_conv = model(input)

	#Add the fully-connected layers 
	x = Flatten(name='flatten')(output_vgg16_conv)
	x = Dense(4096, activation='relu', name='fc1')(x)
	x = Dense(4096, activation='relu', name='fc2')(x)
	x = Dense(2, activation='softmax', name='predictions')(x)

	#Create your own model 
	my_model = Model(inputs=input, outputs=x)

	return my_model
