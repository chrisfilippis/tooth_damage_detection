import imgaug
import imgaug.augmenters as iaa

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.

# es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1)

# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)

def schedule1(model, data_train, data_val, cfg):
    
    augmentation = iaa.OneOf([
        imgaug.augmenters.Fliplr(1.0),
        imgaug.augmenters.Flipud(1.0)
    ])

    model.train(data_train, data_val,
                learning_rate=cfg.LEARNING_RATE,
                epochs=60,
                layers='heads',
                augmentation=augmentation)

    model.train(data_train, data_val,
                learning_rate=cfg.LEARNING_RATE,
                epochs=110,
                layers='all',
                augmentation=augmentation)


def schedule2(model, data_train, data_val, cfg):
    
    augmentation = iaa.OneOf([
        imgaug.augmenters.Fliplr(1.0),
        imgaug.augmenters.Flipud(1.0)
    ])

    model.train(data_train, data_val,
                learning_rate=cfg.LEARNING_RATE,
                epochs=60,
                layers='heads',
                augmentation=augmentation)

    model.train(data_train, data_val,
                learning_rate=cfg.LEARNING_RATE,
                epochs=100,
                layers='4+',
                augmentation=augmentation)
    
    model.train(data_train, data_val,
                learning_rate=cfg.LEARNING_RATE,
                epochs=120,
                layers='all',
                augmentation=augmentation)

    model.train(data_train, data_val,
                learning_rate=cfg.LEARNING_RATE/10,
                epochs=140,
                layers='all',
                augmentation=augmentation)

def schedule3(model, data_train, data_val, cfg):
    
    augmentation = iaa.OneOf([
        imgaug.augmenters.Fliplr(1.0),
        imgaug.augmenters.Flipud(1.0)
    ])

    model.train(data_train, data_val,
                learning_rate=cfg.LEARNING_RATE,
                epochs=20,
                layers='heads',
                augmentation=augmentation)

    model.train(data_train, data_val,
                learning_rate=cfg.LEARNING_RATE,
                epochs=40,
                layers='4+',
                augmentation=augmentation)
    
    model.train(data_train, data_val,
                learning_rate=cfg.LEARNING_RATE,
                epochs=80,
                layers='all',
                augmentation=augmentation)
