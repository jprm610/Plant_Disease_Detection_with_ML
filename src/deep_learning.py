from pathlib import Path
import os

class DeepLearning:
    def __init__(self, source_path: Path, export_path: Path, epochs=100, early_stopping=5) -> None:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable warnings

        self.source_path = source_path
        self.export_path = export_path
        self.epochs = epochs
        self.early_stopping = early_stopping

        # Ensure the export path exists
        self.export_path.mkdir(parents=True, exist_ok=True)

    def main(self):
        print("Training deep learning model...")
        train_df, val_df = self.split_df()
        train_generator, val_generator = self.create_image_data_generators(train_df, val_df)
        print("\tImage generators created!")
        model, earlyStopping = self.define_model(self.early_stopping)

        print("\tFitting model...")
        steps_per_epoch = max(1, train_generator.samples // train_generator.batch_size)
        validation_steps = max(1, val_generator.samples // val_generator.batch_size)

        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=[earlyStopping]
        )

        print("\tModel fitted successfully!")

        self.output_and_export_results(model, train_generator, val_generator)
        self.plot_history(history)

    def split_df(self):
        import pandas as pd
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(f"{self.source_path.parent}/df.csv")

        # Split the dataset into training and validation sets
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

        return train_df, val_df

    def create_image_data_generators(self, train_df, val_df):
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # Create ImageDataGenerators
        train_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)

        # Create data generators
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            directory=self.source_path,
            x_col='image_id',  # Column with image filenames
            y_col='label',     # Column with class labels
            target_size=(224, 224),  # Resize images
            batch_size=64,
            class_mode='categorical'  # Use 'categorical' for multi-class classification
        )

        val_generator = val_datagen.flow_from_dataframe(
            dataframe=val_df,
            directory=self.source_path,
            x_col='image_id',
            y_col='label',
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical'
        )

        return train_generator, val_generator

    def define_model(self, early_stopping):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.regularizers import l2

        earlyStopping = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=early_stopping,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=0,
        )
        
        model = Sequential([
            Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.5),
            Dense(4, activation='softmax')  # 4 classes
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        
        return model, earlyStopping

    def output_and_export_results(self, model, train_generator, val_generator):
        val_loss, val_acc = model.evaluate(val_generator, verbose=2)
        train_loss, train_acc = model.evaluate(train_generator, verbose=2)

        with open(f"{self.export_path.as_posix()}/results.txt", 'w+') as file:
            file.write("\tRESULTS:\n")
            file.write(f"\t\tValidation accuracy: {val_acc}\n")
            file.write(f"\t\tValidation loss: {val_loss}\n")
            file.write(f"\t\tTraining accuracy: {train_acc:.4f}\n")
            file.write(f"\t\tTraining loss: {train_loss:.4f}\n")

            file.seek(0)

            for line in file.readlines():
                print(line, end='')

        model.save(f"{self.export_path.as_posix()}/model.keras")

    def plot_history(self, history):
        from matplotlib import pyplot as plt

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'b', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'b', label='Training Loss')
        plt.plot(epochs, val_loss, 'r', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.savefig(f"{self.export_path.as_posix()}/history.png")

        plt.show()
