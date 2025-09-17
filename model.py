from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
# choice of the network scale should always be a trade-off between performance and speed.

def build_SRCNN():
  model=Sequential()
  model.add(Conv2D(64,(11,11),activation='relu', padding='same',input_shape=(None, None, 3)))
  model.add(Conv2D(32,(1,1),activation='relu', padding='same'))
  model.add(Conv2D(3,(7,7),activation='relu', padding='same'))

  return model

