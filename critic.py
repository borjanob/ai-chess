from tensorflow.keras.layers import Input, Dense, Concatenate, Conv2D, Flatten, MaxPooling2D, Dropout, ReLU
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MSE
from tensorflow.keras import Model
from tensorflow import reduce_mean, convert_to_tensor, squeeze, float32, GradientTape


class Critic(Model):
    

    """
    Defines a critic neural net model used in actor-critic reinforcement learning algorithms
    
    
    
    """

    def __init__(self):
        super(Critic,self).__init__()


