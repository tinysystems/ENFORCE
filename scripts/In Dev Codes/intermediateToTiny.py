import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 1. Set random seed
seed = 22
tf.random.set_seed(seed)
np.random.seed(seed)

# 2. Load dataset
dataset = np.load("librispeech-train-100-clean-mfe-1sec.npz")
samples = dataset['features'][:, :40, :]
classes = dataset['speaker_labels'].astype(int)

# 3. Filter classes with sufficient samples
threshold = 100
counts = np.unique(classes, return_counts=True)[1]
keep_classes = np.where(counts >= threshold)[0]

filtered_samples, filtered_classes = [], []
for i in range(len(classes)):
    if classes[i] in keep_classes:
        filtered_samples.append(samples[i])
        filtered_classes.append(classes[i])

filtered_samples = np.array(filtered_samples)
filtered_classes = np.array(filtered_classes)

# 4. Normalize class indices
unique_classes = np.unique(filtered_classes)
class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
filtered_classes = np.array([class_mapping[cls] for cls in filtered_classes])

# 5. Split into train/val/test
X_train, X_temp, y_train, y_temp = train_test_split(
    filtered_samples, filtered_classes, test_size=0.3, random_state=seed)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=seed)

# Both teacher and student use flattened input (1600,)
X_train_flat = X_train.reshape(X_train.shape[0], -1)  # Shape: (N, 1600)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print(f"Input shape for both models: {X_train_flat.shape}")
print(f"Flattened from original: (N, 40, 40) -> (N, {X_train_flat.shape[1]})")

# 6. Load the complex teacher model (your sparse convolution model)
teacher_model = tf.keras.models.load_model("cnn-librispeech-classifier.h5")
print("Teacher Model (Complex Sparse Convolution):")
teacher_model.summary()

# 7. Target Student Model - EXACTLY as you specified
def build_student_model(input_shape, num_classes=94):
    """Your exact target architecture - simple 3-layer dense network"""
    inputs = tf.keras.Input(shape=input_shape, name="input")  # Already flattened input (1600,)
    
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_1")(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_2")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Dense(256, activation="relu", name="dense_3")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(x)
    
    return tf.keras.Model(inputs, output, name="student_dnn")

# 8. Knowledge Distillation for Complex-to-Simple Model Compression
class ModelCompressionDistiller(tf.keras.Model):
    """
    Distills complex sparse convolution model into simple dense network
    """
    def __init__(self, student, teacher, alpha=0.8, temperature=5.0):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.alpha = alpha  # Higher weight for distillation (complex->simple needs more guidance)
        self.temperature = temperature  # Higher temperature for better knowledge transfer
        
        # Metrics
        self.distillation_loss_tracker = tf.keras.metrics.Mean(name="distill_loss")
        self.student_loss_tracker = tf.keras.metrics.Mean(name="student_loss")
        self.student_accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.distillation_loss_tracker,
            self.student_loss_tracker,
            self.student_accuracy_tracker
        ]
        
    def compile(self, optimizer, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.student_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.distillation_loss_fn = tf.keras.losses.KLDivergence()
        
    def train_step(self, data):
        x, y = data  # Both models use same input now
        
        with tf.GradientTape() as tape:
            # Get predictions from both models
            teacher_predictions = self.teacher(x, training=False)
            student_predictions = self.student(x, training=True)
            
            # Apply temperature scaling for knowledge distillation
            teacher_soft = tf.nn.softmax(teacher_predictions / self.temperature)
            student_soft = tf.nn.softmax(tf.math.log(student_predictions + 1e-10) / self.temperature)
            
            # Calculate losses
            distillation_loss = self.distillation_loss_fn(teacher_soft, student_soft)
            student_loss = self.student_loss_fn(y, student_predictions)
            
            # Weighted combination - emphasize distillation for complex->simple transfer
            total_loss = (
                self.alpha * distillation_loss * (self.temperature ** 2) +
                (1 - self.alpha) * student_loss
            )
            
        # Update student model
        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.distillation_loss_tracker.update_state(distillation_loss)
        self.student_loss_tracker.update_state(student_loss)
        self.student_accuracy_tracker.update_state(y, student_predictions)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "distill_loss": self.distillation_loss_tracker.result(),
            "student_loss": self.student_loss_tracker.result(),
            "accuracy": self.student_accuracy_tracker.result()
        }
        
    def test_step(self, data):
        x, y = data  # Both models use same input now
        
        teacher_predictions = self.teacher(x, training=False)
        student_predictions = self.student(x, training=False)
        
        teacher_soft = tf.nn.softmax(teacher_predictions / self.temperature)
        student_soft = tf.nn.softmax(tf.math.log(student_predictions + 1e-10) / self.temperature)
        
        distillation_loss = self.distillation_loss_fn(teacher_soft, student_soft)
        student_loss = self.student_loss_fn(y, student_predictions)
        total_loss = (
            self.alpha * distillation_loss * (self.temperature ** 2) +
            (1 - self.alpha) * student_loss
        )
        
        self.total_loss_tracker.update_state(total_loss)
        self.distillation_loss_tracker.update_state(distillation_loss)
        self.student_loss_tracker.update_state(student_loss)
        self.student_accuracy_tracker.update_state(y, student_predictions)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "distill_loss": self.distillation_loss_tracker.result(),
            "student_loss": self.student_loss_tracker.result(),
            "accuracy": self.student_accuracy_tracker.result()
        }

# 9. Data preparation - simplified since both models use same input
def create_distillation_data(X, y, batch_size=32, shuffle=True):
    """Create dataset - both models use same flattened input"""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=seed)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# 10. Build student model
input_shape = (1600,)  # Flattened input shape for both models
num_classes = len(unique_classes)

student_model = build_student_model(input_shape, num_classes)
print("\nStudent Model (Target Simple Dense Network):")
student_model.summary()

# 11. Initialize distiller with parameters optimized for complex->simple compression
distiller = ModelCompressionDistiller(
    student=student_model,
    teacher=teacher_model,
    alpha=0.8,  # Emphasize distillation for complex->simple transfer
    temperature=5.0  # Higher temperature for better knowledge transfer
)

distiller.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3)  # Slightly higher LR for simple model
)

# 12. Create datasets - both models use same flattened input
train_data = create_distillation_data(X_train_flat, y_train, batch_size=64)
val_data = create_distillation_data(X_val_flat, y_val, batch_size=64, shuffle=False)

# 13. Callbacks optimized for compression task
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy",
        mode='max',
        factor=0.7,
        patience=8,
        verbose=1,
        min_lr=1e-5
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode='max',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_compressed_student.h5",
        monitor="val_accuracy",
        mode='max',
        save_best_only=True,
        verbose=1
    )
]

# 14. Train the compression
print("\n" + "="*60)
print("COMPRESSING COMPLEX MODEL TO SIMPLE DENSE NETWORK")
print("="*60)
print(f"Teacher: Sparse Conv Model ({teacher_model.count_params():,} params)")
print(f"Student: Simple Dense Model ({student_model.count_params():,} params)")
print(f"Compression Ratio: {teacher_model.count_params() / student_model.count_params():.1f}x")
print("="*60)

history = distiller.fit(
    train_data,
    validation_data=val_data,
    epochs=150,
    callbacks=callbacks,
    verbose=1
)

# 15. Save the compressed model
model_name = "compressed_dense_student"
student_model.save(f"{model_name}.h5")
print(f"\nCompressed model saved as {model_name}.h5")

# 16. Convert to TFLite for deployment
def convert_to_tflite(keras_model, output_filename):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    os.makedirs('models', exist_ok=True)
    with open(os.path.join('models', output_filename), 'wb') as f:
        f.write(tflite_model)
    
    return len(tflite_model)

tflite_size = convert_to_tflite(student_model, f'{model_name}.tflite')

# 17. Comprehensive evaluation
print("\n" + "="*60)
print("COMPRESSION RESULTS")
print("="*60)

# Evaluate on test set
test_data = create_distillation_data(X_test_flat, y_test, batch_size=64, shuffle=False)
test_results = distiller.evaluate(test_data, verbose=0)

print(f"Final Test Results:")
print(f"  Total Loss: {test_results[0]:.4f}")
print(f"  Distillation Loss: {test_results[1]:.4f}")
print(f"  Student Loss: {test_results[2]:.4f}")
print(f"  Student Accuracy: {test_results[3]:.4f}")

# 18. Compare teacher vs compressed student
print(f"\nPerformance Comparison (1000 test samples):")
teacher_pred = teacher_model.predict(X_test_flat[:1000], verbose=0)
student_pred = student_model.predict(X_test_flat[:1000], verbose=0)

teacher_acc = np.mean(np.argmax(teacher_pred, axis=1) == y_test[:1000])
student_acc = np.mean(np.argmax(student_pred, axis=1) == y_test[:1000])

print(f"  Teacher (Complex) Accuracy: {teacher_acc:.4f}")
print(f"  Student (Simple) Accuracy: {student_acc:.4f}")
print(f"  Performance Retention: {(student_acc/teacher_acc)*100:.1f}%")

# 19. Architecture comparison
print(f"\nArchitecture Comparison:")
print(f"  Teacher: Sparse Conv + SoftMax Pooling + Dense")
print(f"  Student: Simple 3-Layer Dense Network")
print(f"  Parameter Reduction: {((teacher_model.count_params() - student_model.count_params()) / teacher_model.count_params()) * 100:.1f}%")

# 20. Model size analysis
def get_model_size(model_path):
    return os.path.getsize(model_path) / (1024 * 1024)

student_size = get_model_size(f"{model_name}.h5")
tflite_size_mb = tflite_size / (1024 * 1024)

print(f"\nModel Size:")
print(f"  Compressed Student (.h5): {student_size:.2f} MB")
print(f"  TFLite Model: {tflite_size_mb:.2f} MB")

print("\n" + "="*60)
print("MODEL COMPRESSION COMPLETED SUCCESSFULLY!")
print("Complex sparse convolution model compressed to simple 3-layer dense network")
print("="*60)
