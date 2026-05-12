import tensorflow as tf

# Ucitavanje dataset-a
train_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset_n",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=32
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset_n",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=32
)

# Ispis klasa
print("Klase:", train_dataset.class_names)
print("Dataset uspesno ucitan!")

# import matplotlib.pyplot as plt
#
# class_names = train_dataset.class_names
#
# plt.figure(figsize=(10,10))
#
# for images, labels in train_dataset.take(1):
#     for i in range(9):
#         ax = plt.subplot(3,3,i+1)
#
#         plt.imshow(images[i].numpy().astype("uint8"))
#
#         plt.title(class_names[labels[i]])
#
#         plt.axis("off")
#
# plt.show()


normalization_layer = tf.keras.layers.Rescaling(1./255)

train_dataset = train_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

validation_dataset = validation_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu',
                           input_shape=(128,128,3)),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()