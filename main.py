import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

path = 'dataset'
path_imgs = list(glob.glob(path + '/**/*.jpg'))

for dirname, _, filenames in os.walk('dataset'):
    for filename in filenames:
        os.path.join(dirname, filename)
class_names = sorted(os.listdir(path))

n_classes = len(class_names)
print(f"Total Num of Classes : {n_classes}")

class_dis = [len(os.listdir(path + "/" + name)) for name in class_names]
print(f"Total Num of Images : {sum(class_dis)}")

for i in range(n_classes):
    print(f"{class_names[i]} : {class_dis[i]}")

# Plotting the distribution of classes pie chart
fig1, ax1 = plt.subplots()
ax1.pie(class_dis, labels=class_names, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')
plt.show()

# Plotting the distribution of classes bar chart
plt.figure(figsize=(10, 8))
sns.barplot(x=class_names, y=class_dis)
plt.axhline(np.mean(class_dis), alpha=0.5, linestyle='--', color='k', label="Mean")
plt.title("Class Distribution")
plt.legend(fontsize=15)
plt.show()

labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], path_imgs))
file_path = pd.Series(path_imgs, name='File_Path').astype(str)
labels = pd.Series(labels, name='Labels')
data = pd.concat([file_path, labels], axis=1)
data = data.sample(frac=1).reset_index(drop=True)
data.head()

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 7), subplot_kw={'xticks': [], 'yticks': []})
for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(data.File_Path[i]))
    ax.set_title(data.Labels[i])
plt.tight_layout()
plt.show()

counts = data.Labels.value_counts()
sns.barplot(x=counts.index, y=counts)
plt.xlabel('Labels')
plt.ylabel('Count')
plt.xticks(rotation=50)

train_df, test_df = train_test_split(data, test_size=0.3, random_state=2)
print(f"Train Data : {train_df.shape}")
print(f"Test Data : {test_df.shape}")


def gen(pre, train, test):
    train_datagen = ImageDataGenerator(preprocessing_function=pre, validation_split=0.2)
    test_datagen = ImageDataGenerator(preprocessing_function=pre)
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train, x_col='File_Path', y_col='Labels', target_size=(100, 100), class_mode='categorical',
        batch_size=32, shuffle=True, seed=0, subset='training', rotation_range=30, zoom_range=0.15,
        width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
    valid_gen = train_datagen.flow_from_dataframe(
        dataframe=train, x_col='File_Path', y_col='Labels', target_size=(100, 100), class_mode='categorical',
        batch_size=32, shuffle=False, seed=0, subset='validation', rotation_range=30, zoom_range=0.15,
        width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test, x_col='File_Path', y_col='Labels', target_size=(100, 100), color_mode='rgb',
        class_mode='categorical', batch_size=32, verbose=0, shuffle=False)
    return train_gen, valid_gen, test_gen


def func(name_model):
    pre_model = name_model(input_shape=(100, 100, 3), include_top=False, weights='imagenet', pooling='avg')
    pre_model.trainable = False
    inputs = pre_model.input
    x = Dense(100, activation='relu')(pre_model.output)
    x = Dense(100, activation='relu')(x)
    outputs = Dense(11, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    my_callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='auto')]
    return model, my_callbacks


def plot(history, test_gen, train_gen, model):
    # Plotting Accuracy, val_accuracy, loss, val_loss
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax = ax.ravel()

    for i, met in enumerate(['accuracy', 'loss']):
        ax[i].plot(history.history[met])
        ax[i].plot(history.history['val_' + met])
        ax[i].set_title('Model {}'.format(met))
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(met)
        ax[i].legend(['Train', 'Validation'])

    # Predict Data Test
    pred = model.predict(test_gen)
    pred = np.argmax(pred, axis=1)
    labels = (train_gen.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    pred = [labels[k] for k in pred]

    # Classification report
    cm = confusion_matrix(test_df.Labels, pred)
    clr = classification_report(test_df.Labels, pred)
    print(clr)
    # Display 6 picture of the dataset with their labels
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 8),
                             subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(test_df.File_Path.iloc[i + 1]))
        ax.set_title(f"True: {test_df.Labels.iloc[i + 1]}\nPredicted: {pred[i + 1]}")
    plt.tight_layout()
    plt.show()
    return history


def result_test(test, model_use):
    results = model_use.evaluate(test, verbose=0)
    print("Test Loss: {:.5f}".format(results[0]))
    print("Test Accuracy: {:.2f}%".format(results[1] * 100))
    return results


# ResNet50
ResNet_pre = preprocess_input
train_gen_ResNet, valid_gen_ResNet, test_gen_ResNet = gen(ResNet_pre, train_df, test_df)
ResNet_model, callback = func(ResNet50)
history = ResNet_model.fit(train_gen_ResNet, validation_data=valid_gen_ResNet, epochs=100, callbacks=callback,
                           verbose=0)
history_ResNet = plot(history, test_gen_ResNet, train_gen_ResNet, ResNet_model)
result_ResNet = result_test(test_gen_ResNet, ResNet_model)

# MobileNet
MobileNet_pre = preprocess_input
train_gen_MobileNet, valid_gen_MobileNet, test_gen_MobileNet = gen(MobileNet_pre, train_df, test_df)
MobileNet_model, callback = func(MobileNet)
history = MobileNet_model.fit(train_gen_MobileNet, validation_data=valid_gen_MobileNet, epochs=100, callbacks=callback,
                              verbose=0)
history_MobileNet = plot(history, test_gen_MobileNet, train_gen_MobileNet, MobileNet_model)
result_MobileNet = result_test(test_gen_MobileNet, MobileNet_model)

# VGG19
VGG19_pre = preprocess_input
train_gen_VGG19, valid_gen_VGG19, test_gen_VGG19 = gen(VGG19_pre, train_df, test_df)
VGG19_model, callback = func(VGG19)
history = VGG19_model.fit(train_gen_VGG19, validation_data=valid_gen_VGG19, epochs=100, callbacks=callback, verbose=0)
history_VGG19 = plot(history, test_gen_VGG19, train_gen_VGG19, VGG19_model)
result_VGG19 = result_test(test_gen_VGG19, VGG19_model)

# Xception
Xception_pre = preprocess_input
train_gen_Xception, valid_gen_Xception, test_gen_Xception = gen(Xception_pre, train_df, test_df)
Xception_model, callback = func(Xception)
history = Xception_model.fit(train_gen_Xception, validation_data=valid_gen_Xception, epochs=100, callbacks=callback,
                             verbose=0)
history_Xception = plot(history, test_gen_Xception, train_gen_Xception, Xception_model)
result_Xception = result_tresult_Xception = result_test(test_gen_Xception, Xception_model)

# InceptionResNetV2
IRNV2_pre = preprocess_input
train_gen_IRNV2, valid_gen_IRNV2, test_gen_IRNV2 = gen(IRNV2_pre, train_df, test_df)
IRNV2_model, callback = func(InceptionResNetV2)
history = IRNV2_model.fit(train_gen_IRNV2, validation_data=valid_gen_IRNV2, epochs=100, callbacks=callback, verbose=0)
history_IRNV2 = plot(history, test_gen_IRNV2, train_gen_IRNV2, IRNV2_model)
result_IRNV2 = result_tresult_IRNV2 = result_test(test_gen_IRNV2, IRNV2_model)

# VGG16
vgg_pre = preprocess_input
train_gen_VGG, valid_gen_VGG, test_gen_VGG = gen(vgg_pre, train_df, test_df)
model_VGG16, callback = func(VGG16)
history = model_VGG16.fit(train_gen_VGG, validation_data=valid_gen_VGG, epochs=100, callbacks=callback, verbose=0)
history = plot(history, test_gen_VGG, train_gen_VGG, model_VGG16)
result_VGG16 = result_test(test_gen_VGG, model_VGG16)

# ResNet101
ResNet101_pre = preprocess_input
train_gen_ResNet101, valid_gen_ResNet101, test_gen_ResNet101 = gen(ResNet101_pre, train_df, test_df)
model_ResNet101, callback = func(ResNet101)
history = model_ResNet101.fit(train_gen_ResNet101, validation_data=valid_gen_ResNet101, epochs=100, callbacks=callback,
                              verbose=0)
history = plot(history, test_gen_ResNet101, train_gen_ResNet101, model_ResNet101)
result_ResNet101 = result_test(test_gen_ResNet101, model_ResNet101)

# DenseNet201
DenseNet201_pre = preprocess_input
train_gen_DenseNet201, valid_gen_DenseNet201, test_gen_DenseNet201 = gen(DenseNet201_pre, train_df, test_df)
model_DenseNet201, callback = func(DenseNet201)
history = model_DenseNet201.fit(train_gen_DenseNet201, validation_data=valid_gen_DenseNet201, epochs=100,
                                callbacks=callback, verbose=0)
history = plot(history, test_gen_DenseNet201, train_gen_DenseNet201, model_DenseNet201)
result_DenseNet201 = result_test(test_gen_DenseNet201, model_DenseNet201)

output = pd.DataFrame(
    {'Model': ['ResNet50', 'MobileNet', 'VGG19', 'Xception', 'InceptionResNetV2', 'VGG16', 'ResNet101', 'DenseNet201'],
     'Accuracy': [result_ResNet[1], result_MobileNet[1], result_VGG19[1], result_Xception[1],
                  result_IRNV2[1], result_VGG16[1], result_ResNet101[1], result_DenseNet201[1]]})

plt.figure(figsize=(12, 7))
plots = sns.barplot(x='Model', y='Accuracy', data=output)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'), (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center',
                   va='center', size=15, xytext=(0, 8), textcoords='offset points')
plt.xlabel("Models", size=14)
plt.ylabel("Accuracy", size=14)
plt.show()
