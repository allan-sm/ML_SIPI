# pyton version 3.6.5

# Crear red neuronal
# h5py 2.10.0
# tensorflow 1.15.0
# Keras 2.3.1
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
# from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Conv2D, MaxPooling2D, Conv1D
from keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

# Manejo de datos para estandarizar datos al modelo
# sklearn  0.0.post5
# scikit-learn   0.22.1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump, load
from sklearn.model_selection import train_test_split

# Manejo de datos y documentos
# pandas 1.1.5
import pandas

# Manejo matematico y numerico
# numpy  1.19.5
import numpy as np

# Graficas
# matplotlib  3.3.4
import matplotlib.pyplot as plt

# Importar dataset
dataset = pandas.read_csv("Light_sample.csv", header=None)

# Otener parametros de linea 1 en un arreglo y valores numericos en forma de matriz
def get_values_and_header(csv_matrix):
    csv_matrix=csv_matrix.values
    matrix_values=csv_matrix[1:,0:]
    matrix_values=np.asfarray(matrix_values,float)
    vector_header=csv_matrix[0,0:]
    return matrix_values, vector_header

dataset, vector_header=get_values_and_header(dataset)

# Definicion de parametros de entrada y de salida
features_amount=8
outputs_amount=12
features=dataset[0,0:features_amount]

# Funciones para separar datos considerandotodo el barrido de frequencia
def separador_de_features(features_amount,dataset,features):
    contafeatures=0
    features=np.reshape(features, (1, features_amount))
    print(features)

    for i in range(0,len(dataset)):
        for j in range(0,features_amount):
            if features[contafeatures][j]!=dataset[i][j]:
                son_iguales=False
                break
            else:
                son_iguales=True
        if son_iguales==False:
            dataset_sub_i_2d=np.reshape(dataset[i,0:features_amount], (1, features_amount))
            features=np.concatenate((features,dataset_sub_i_2d))
            contafeatures=contafeatures+1
    return features

features=separador_de_features(features_amount,dataset,features)

# Separar datos en montos para entrenamiento y para testeo

def train_test_full(features_amount,dataset,x_test,outputs_amount):
        conta_iguales=0
        data_test_full=np.ones((1,features_amount+1+outputs_amount))
        print(data_test_full)

        for i in range(0,len(x_test)):
            #rows_dataset=len(dataset)
            for j in range(0,len(dataset)):
                    for k in range(0,features_amount):
                        if x_test[i][k]!=dataset[j-conta_iguales][k]:
                            son_iguales=False
                            break
                        else:
                            son_iguales=True
                    if son_iguales==True:
                        dataset_sub_i_2d=np.reshape(dataset[j-conta_iguales,0:features_amount+1+outputs_amount], (1, features_amount+1+outputs_amount))
                        data_test_full=np.concatenate((data_test_full,dataset_sub_i_2d))
                        dataset=np.delete(dataset, j-conta_iguales, axis=0)
                        conta_iguales=conta_iguales+1
            print(str((i+1)/len(x_test)*100)+"%")
            conta_iguales=0
        data_test_full=data_test_full[1:len(data_test_full),:]
        return data_test_full, dataset

x_train, x_test = train_test_split(features, test_size=0.15, random_state=1)

vector_header=np.reshape(vector_header,(1,len(vector_header)))

data_test_full, dataset=train_test_full(features_amount,dataset,x_test,outputs_amount)

data_test_full=np.append(vector_header,data_test_full,axis=0)
dataset=np.append(vector_header,dataset,axis=0)

# Crear archivos de datos para entrenamiento y testeo separados
np.savetxt("testing_set.csv",data_test_full,delimiter=",",fmt='%s')
np.savetxt("training_set.csv",dataset,delimiter=",",fmt='%s')


# Separar datos para validacion (solo se divide en 2)
dataset = pandas.read_csv("training_set.csv", header=None)
dataset, vector_header=get_values_and_header(dataset)

#dataset= np.array([[1,2,3,1,3],[1,2,3,2,4],[1,2,7,1,5],[1,2,7,2,6],[1,2,1,1,7],[1,2,1,2,17],[1,2,1,2,17],[1,44,1,1,7],[1,44,1,2,17]])
#x_test=np.array([[1,2,3],[1,2,1]])
features_amount=8
outputs_amount=12

features=dataset[0,0:features_amount]

features=separador_de_features(features_amount,dataset,features)

x_train, x_test = train_test_split(
    features, test_size=3/17, random_state=1)

# Tambien se podria seccionar al azar
#np.savetxt("training_set_re_im_caso_simple_random_state_1.csv",x_train,delimiter=",")
#np.savetxt("testing_set_re_im_caso_simple_random_state_1.csv",x_test,delimiter=",")

vector_header=np.reshape(vector_header,(1,len(vector_header)))

data_test_full, dataset=train_test_full(features_amount,dataset,x_test,outputs_amount)

data_test_full=np.append(vector_header,data_test_full,axis=0)
dataset=np.append(vector_header,dataset,axis=0)

np.savetxt("validation_set.csv",data_test_full,delimiter=",",fmt='%s')
np.savetxt("training_set_divided.csv",dataset,delimiter=",",fmt='%s')



# Cargar dataset ya dividido
dataframe = pandas.read_csv(r"training_set_divided.csv", header=None)
dataframe = dataframe.values
dataframe = np.asfarray(dataframe[1:, 0:], float)

# dataframe=np.delete(dataframe,7,1)
x_train = dataframe[:, 0:8]
y_train = dataframe[:, 8:]
print(x_train.shape)
print(y_train.shape)

dataframe = pandas.read_csv(r"testing_set.csv", header=None)
dataframe = dataframe.values
dataframe = np.asfarray(dataframe[1:, 0:], float)

# dataframe=np.delete(dataframe,7,1)
x_test = dataframe[:, 0:8]
y_test = dataframe[:, 8:]

dataframe = pandas.read_csv(r"validation_set.csv", header=None)
dataframe = dataframe.values
dataframe = np.asfarray(dataframe[1:, 0:], float)

# dataframe=np.delete(dataframe,7,1)
x_val = dataframe[:, 0:8]
y_val = dataframe[:, 8:]


# Crear modelo
model = Sequential()
model.add(Dense(50, activation='relu', input_dim=x_train.shape[1]))
model.add(Dense(50, activation='relu'))
model.add(Dense(y_train.shape[1]))

# Estandarizar datos
standar_1=StandardScaler().fit(x_train)
x_train=standar_1.transform(x_train)
x_val=standar_1.transform(x_val)
x_test=standar_1.transform(x_test)
# Guardar estandarizacion en documento
dump(standar_1, 'standarization.bin', compress=True)

# Setup del modelo
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Entrenar el modelo
model.fit(x_train, y_train, epochs=3, verbose=2, validation_data=(x_val, y_val))

# Evaluar el modelo
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

# Guardar modelo entrenado
model.save("model.h5")

# Cargar modelo y estandarizacion
model = load_model('model.h5')
standar_1=load('standarization.bin')
model.summary()
