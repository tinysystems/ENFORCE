
import math
from tensorflow.keras import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D, InputLayer, Reshape 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import CosineSimilarity
import tensorflow.keras.backend as K

def cosine_similarity_loss(y_true, y_pred):
    y_true=K.l2_normalize(y_true, axis=-1)
    y_pred=K.l2_normalize(y_pred, axis=-1)
    return 1-K.sum(y_true*y_pred, axis=-1)

def combined_similarity_loss(y_true, y_pred, alpha=1):
    cosine_loss=cosine_similarity_loss(y_true, y_pred)
    l2_loss=tf.norm(y_true-y_pred, axis=-1)
    combined_loss=(alpha*l2_loss)+((1.0-alpha)*cosine_loss)
    return tf.reduce_mean(combined_loss)

def distillation_loss(y_true, y_pred, teacher_targets, alpha):
    "Combines label loss and soft label distillation loss"
    #student_loss=tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
    kl_divergence=tf.keras.losses.KLDivergence()(teacher_targets, y_pred)
    return kl_divergence

class CompleteTransitionToFC:
    def __init__(self, teacher_model, input_shape):
        "Class Initialization"
        print("="*80)
        print("INITIALIZING DISTILLATION CLASS")
        print("="*80)
        self.teacher_model=teacher_model
        self.input_shape=input_shape
        self.input_size=np.prod(input_shape)
        self.num_classes=-1
        self.teacher_layers={}
        self.student_model=None
        self.layer_mapping={}
        print(f"Teacher model provided: {teacher_model is not None}")
        print(f"Input shape: {self.input_shape}")
        print(f"Flattened input: {self.input_size}")
        print("\nStarting teacher model analysis")
        self.analyze_teacher_model()
        print("Initialization completed successfully!")
    
    def analyze_teacher_model(self):
        "Extract layer into from teacher"
        print("\n"+"="*80)
        print("ANALYZING TEACHER MODEL ARCHITECTURE")
        print("="*80)
        print("Teacher model summary:")
        try:
            self.teacher_model.summary()
            print("Model summary displayed successfully!")
        except Exception as e:
            print(f"Error displaying teacher model: {e}")
            
        print("\nVerification if there are only supported layers\n")
        print(f"Model contains {len(self.teacher_model.layers)} layers")
        
        layer_outputs=[]
        layer_names=[]

        for i, layer in enumerate(self.teacher_model.layers):
            layer_type=type(layer).__name__
            print(f" Layer {i:2d}: {layer_type:20s} - {layer.name}")
            if isinstance(layer, (InputLayer, Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout)):
                layer_outputs.append(layer.output)
                layer_names.append(f"{layer.name}_{i}")
            else:
                print(f"Error at Layer {i}")
                return
        
        print("Neural Network can be processed by the framework")
        print("Setting all the layers as outputs...")
        try:
            self.layerExtractor=Model(inputs=self.teacher_model.input, outputs=layer_outputs)
            print("Layer Extractor generated successfully")
        except Exception as e:
            print(f"Error generating Layer Extractor: {e}")
            return
        
        self.layer_names=layer_names
        self.extract_layer_details()
        self.create_layer_mapping()
        for layer in self.teacher_model.layers:
            layer.trainable=False
        
    def extract_layer_details(self):
        "INFO Extraction of layer"
        print("\n"+"="*50)
        print("DETAILED INFO EXTRACTING")
        print("="*50)
        
        teacher_layers=[]
        for i, layer in enumerate(self.teacher_model.layers):
            if hasattr(layer, 'output'):
                output_shape=layer.output.shape
                output_size=np.prod(output_shape[1:]) if len(output_shape)>1 else 1
                layer_info={
                    'index': i,
                    'name': layer.name,
                    'type': type(layer).__name__,
                    'layer_obj': layer,
                    'output_shape': output_shape,
                    'output_size': output_size
                }
                if isinstance(layer, MaxPooling2D):
                    layer_info['pool_size']=layer.pool_size
                    layer_info['strides']=layer.strides
                elif isinstance(layer, Conv2D):
                    layer_info['filters']=layer.filters
                    layer_info['kernel_size']=layer.kernel_size
                    layer_info['strides']=layer.strides
                    layer_info['padding']=layer.padding
                    
                teacher_layers.append(layer_info)
                print(f"    Layer {i:2d}: {layer_info['name']:30s} | Type: {layer_info['type']:25s} | Shape: {str(layer_info['output_shape']):20s} | Size: {layer_info['output_size']:6d}")
        self.teacher_layers=teacher_layers
    
    def create_layer_mapping(self):
        "Create map between teacher CNN and student Dense"
        print("\n"+"-"*0)
        print("CREATING LAYER MAPPING FOR DISTILLATION")
        print("-"*80)
        
        student_idx=1 #Start layer after flatten layer
        for teacher_layer in self.teacher_layers:
            layer_type=teacher_layer['type']
            teacher_idx=teacher_layer['index']
            if layer_type in ['Conv2D', 'Dense']:
                #if layer_type=='Conv2D' and teacher_layer['layer_obj'].strides!=(1,1):
                #    student_idx+=2
                self.layer_mapping[teacher_idx]=student_idx
                print(f"    Mapping: Teacher Layer {teacher_idx} ({teacher_layer['name']}) -> Student Dense Layer {student_idx}")
                student_idx+=2
            elif layer_type in ['MaxPooling2D']:
                student_idx+=3
            else:
                self.layer_mapping[teacher_idx]=-1
                print(f"    Mapping: Teacher Layer {teacher_idx} ({teacher_layer['name']}) -> None")
    
    def create_dense_student(self, dropout_rate=0.3):
        "Create dense neural network with number of layers equal to CNN"
        print("\n"+"-"*80)
        print("CREATING STUDENT DENSE NETWORK")
        print("-"*80)
        
        input_layer=Input(shape=(self.input_size,), name="input")
        x=input_layer
        
        for i, layer in enumerate(self.teacher_layers):
            output_size=layer['output_size']
            output_shape=layer['output_shape']
            match layer['type']:
                case 'InputLayer':
                    pass
                case 'Dense':
                    if i==len(self.teacher_layers)-1:
                        x=Dense(output_size, activation="softmax", name=f"dense_{i}_softmax")(x)
                        self.num_classes=output_size
                    else:
                        x=Dense(output_size, activation="relu", name=f"dense_{i}_relu")(x)
                        x=Dropout(dropout_rate, name=f"dropout_{i}")(x)
                case 'Conv2D':
                    #if layer['layer_obj'].strides==(1,1):
                    #if layer['strides']==(1,1):
                    x=Dense(output_size, activation="relu", name=f"dense_{i}_conv")(x)
                    x=Dropout(dropout_rate, name=f"dropout_{i}")(x)
                    """
                    else:
                        print(output_size)
                        prev_layer=self.teacher_layers[i-1]
                        start_shape=prev_layer['output_shape']
                        if len(start_shape)>=4:
                            _, h_in, w_in, c_in=start_shape
                        else:
                            h_in, w_in, c_in=start_shape
                        c_in*=2
                        intermediate=(h_in, w_in, c_in)
                        
                        print(f"{h_in}, {w_in}, {c_in}")
                        x=Dense(np.prod(intermediate), activation="relu", name=f"dense_{i}_conv")(x)
                        x=Dropout(dropout_rate, name=f"dropout_{i}")(x)
                        x=Reshape((h_in, w_in, c_in), name=f"reshape_to_pool_{i}")(x)
                        x=SoftMaxPooling2D(pool_size=layer['strides'], alpha=10.0, name=f"soft_max_pool_extra_{i}")(x)
                        x=Flatten(name=f"flatten_{i}")(x)
                    """
                case 'MaxPooling2D':
                    prev_layer=self.teacher_layers[i-1]
                    prev_shape=prev_layer['output_shape']
                    if len(prev_shape)>=4:
                        _, h_in, w_in, c_in=prev_shape
                        _, h_out, w_out, c_out=output_shape
                    else:
                        h_in, w_in, c_in=prev_shape
                        h_out, w_out, c_out=output_shape
                    
                    x=Reshape((h_in, w_in, c_in), name=f"reshape_to_pool_{i}")(x)
                    x=SoftMaxPooling2D(target_output_shape=(h_out, w_out), alpha=10.0, name=f"soft_max_pool_{i}")(x)
                    x=Flatten(name=f"flatten_{i}")(x)
                    #x=Dense(output_size, activation="relu", name=f"dense_{i}_pool")(x)
                    #x=Dropout(dropout_rate, name=f"dropout_{i}")(x)
                #case 'BatchNormalization':
                #    x= BatchNormalization(name=f"bn_{i}", trainable=False)(x)
                case 'Flatten' | 'Dropout':
                    continue
                case _:
                    print(f"Warning: Unsupported layer type {layer['type']}")
        
        student_model=Model(
            inputs=input_layer,
            outputs=x,
            name="FullyConnectedModel"
        )
        
        sample_input = np.zeros(self.input_size)
        sample_input = np.expand_dims(sample_input, axis=0)
        _=student_model(sample_input)
        self.student_model=student_model

        student_model.summary()
        self.save_student_model('distilled_student_model.h5')
        
        return self.student_model
    
    def set_mathematical_weights(self, student_model):
        "Set weights from teacher CNN to student Dense"
        print("\n"+"="*80)
        print("SETTING WEIGHTS WITH MATH METHOD")
        print("="*80)

        current_shape=self.input_shape 
        for layer in self.teacher_layers:
            teacher_idx=layer['index']
            teacher_type=layer['type']
            teacher_layer=layer['layer_obj']
            
            if teacher_idx in self.layer_mapping and self.layer_mapping[teacher_idx]!=-1:
                student_idx=self.layer_mapping[teacher_idx]
                student_layer=student_model.layers[student_idx]
                print(f"    Teacher Layer {teacher_idx} ({teacher_layer.name}) -> Student Layer {student_idx} ({student_layer.name})")
                try:
                    if teacher_type=='Conv2D':
                        dense_weights, biases=self.conv2d_to_dense_weights(teacher_layer, current_shape)
                        student_layer.set_weights([dense_weights, biases])
                        print(f"    Set weights for {student_layer.name}")
                    elif teacher_type=='Dense':
                        student_layer.set_weights(teacher_layer.get_weights())
                        print(f"    Set weights for {student_layer.name}")
                except Exception as e:
                    print(f"    Error setting weights for {teacher_layer.name}: {e}")
            current_shape=layer['output_shape'][1:]
    """
    def set_mathematical_weights(self, student_model):
        print("\n"+"="*80)
        print("SETTING WEIGHTS WITH MATH METHOD")
        print("="*80)
        
        student_layer_idx=1
        current_shape=self.input_shape
        for layer in self.teacher_layers:
            if layer['type'] == 'Conv2D':
                conv_layer=layer['layer_obj']
                try:
                    dense_weights, biases=self.conv2d_to_dense_weights(conv_layer, current_shape)
                    student_dense_layer=student_model.layers[student_layer_idx]
                    student_dense_layer.set_weights([dense_weights, biases])
                    print(f"    Set weights for {student_dense_layer.name}")
                except Exception as e:
                    print(f"    Error setting weights for conv2D layer: {e}")
                student_layer_idx+=2
            elif layer['type'] in ['MaxPooling2D', 'Flatten', 'Dropout']:
                if layer['type']=='MaxPooling2D':
                    student_layer_idx+=2
                elif layer['type']=='Flatten':
                    pass
                else:
                    pass    
            elif layer['type'] in ['Dense']:
                teacher_dense=layer['layer_obj']
                student_dense=student_model.layers[student_layer_idx]
                if teacher_dense.units==student_dense.units:
                    student_dense.set_weights(teacher_dense.get_weights())
                else:
                    print(f"    Skipped {teacher_dense.name} (shape mismatch)")
                student_layer_idx+=2
            #elif layer['type'] in ['BatchNormalization']:
            #    bn_layer=student_model.layers[student_layer_idx]
            #    bn_layer.trainable=False
            #    teacher_weights=layer['layer_obj'].get_weights()
            #    units=current_shape[-1]
            #    if len(teacher_weights[0].shape)==0:
            #        new_weights=[
            #            np.full(units, teacher_weights[0]),
            #            np.full(units, teacher_weights[1]),
            #            np.full(units, teacher_weights[2]),
            #            np.full(units, teacher_weights[3])
            #        ]
            #        bn_layer.set_weights(new_weights)
            #    student_layer_idx+=1
            #    print(f"    Initialized BN layer {bn_layer.name}")
            current_shape=layer['output_shape'][1:]
    """ 
            
    def freeze_all_layers(self):
        "Freeze all layers of the model"
        for layer in self.student_model.layers:
            layer.trainable=False
        print("All student layers frozen")
    
    def unfreeze_layer(self, layer_idx):
        "Unfreeze specific layer"
        if layer_idx<len(self.student_model.layers):
            self.student_model.layers[layer_idx].trainable=True
            print(f"Unfrozen layer {layer_idx}: {self.student_model.layers[layer_idx].name}")
        else:
            print(f"Warning: Layer index {layer_idx} out of range")
    
    def train_layer_sequential(self, teacher_idx, student_idx, X_train, y_train, X_val, y_val, epochs=30, learning_rate=0.001):
        "Train student layer to mimic teacher layer output"
        print(f"\n"+"="*60)
        print(f"TRAINING LAYER {student_idx} THAT MIMICS TEACHER LAYER {teacher_idx}")
        print("="*60)
        
        if not hasattr(self, 'student_model') or self.student_model is None:
            self.student_model=self.create_dense_student()
        
        if not self.student_model.built:
            print("Building Student Model")
            dummy_input = np.zeros(self.input_shape)
            dummy_input=np.expand_dims(dummy_input, axis=0)
        
        self.freeze_all_layers()
        self.unfreeze_layer(student_idx)
        
        teacher_layer_model=Model(
            inputs=self.teacher_model.input,
            outputs=self.teacher_model.layers[teacher_idx].output
        )
        teacher_layer_model.trainable=False
        
        student_intermediate=Model(
            inputs=self.student_model.input,
            outputs=self.student_model.layers[student_idx].output
        )
        
        optimizer=Adam(learning_rate=learning_rate)
        student_intermediate.compile(
            optimizer=optimizer,
            loss=combined_similarity_loss,
            metrics=[CosineSimilarity(name="cos_sim")]
        )
        
        history={
            'loss': [], 
            'cos_sim': [], 
            'val_loss': [], 
            'val_cos_sim': []
        }
        
        #callbacks=[
        #    EarlyStopping(
        #        monitor='val_loss',
        #        patience=10,
        #        restore_best_weights=True,
        #        verbose=1
        #    ),
        #    ReduceLROnPlateau(
        #        monitor='val_loss',
        #        factor=0.5,
        #        patience=5,
        #        min_lr=1e-7,
        #        verbose=1
        #    )
        #]
        try:
            from tqdm.auto import tqdm
            epoch_bar=tqdm(range(epochs), desc='Training Progress')
        except ImportError:
            epoch_bar=range(epochs)
            
        print(f"Training student layer {student_idx}")
        for epoch in epoch_bar:
            train_metrics=self.run_training_epoch(
                student_model=student_intermediate,
                teacher_model=teacher_layer_model,
                X=X_train,
                y=y_train,
                optimizer=optimizer
            )
            
            val_metrics=self.run_validation_epoch(
                student_model=student_intermediate,
                teacher_model=teacher_layer_model,
                X=X_val,
                y=y_val
            )
            
            history['loss'].append(train_metrics['loss'])
            history['cos_sim'].append(train_metrics['cos_sim'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_cos_sim'].append(val_metrics['cos_sim'])
            
            if hasattr(epoch_bar, 'set_postfix'):
                epoch_bar.set_postfix({
                    'loss': f"{train_metrics['loss']:.4f}",
                    'cos_sim': f"{train_metrics['cos_sim']:.4f}",
                    'val_loss': f"{val_metrics['loss']:.4f}",
                    'val_cos': f"{val_metrics['cos_sim']:.4f}"
                })
        try:
            self.plot_training_curves(history, student_idx)
        except ImportError:
            print("Matplotlib not available")
        return history
    
    def run_training_epoch(self, student_model, teacher_model, X, y, optimizer):
        "Run one training epoch"            
        epoch_loss=[]
        epoch_cos=[]
        batch=32
        
        for i in range(0, len(X), batch):
            batch_X_t=X[i:i+batch]
            batch_y=y[i:i+batch]
            batch_X_s=batch_X_t.reshape(batch_X_t.shape[0], -1)
            teacher_layer_obj=teacher_model.layers[-1]
            teacher_strides=teacher_layer_obj.strides if isinstance(teacher_layer_obj, tf.keras.layers.Conv2D) else (1,1)
            
            with tf.GradientTape() as tape:
                teacher_features=teacher_model(batch_X_t, training=False)
                student_features=student_model(batch_X_s, training=True)
                teacher_flat=tf.reshape(teacher_features, [batch_X_t.shape[0], -1])
                student_flat=tf.reshape(student_features, [batch_X_s.shape[0], -1])
                total_loss=tf.reduce_mean(combined_similarity_loss(teacher_flat, student_flat))
            gradients=tape.gradient(total_loss, student_model.trainable_variables)
            if gradients:
                optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))
            epoch_loss.append(float(total_loss))
            epoch_cos.append(float(1-total_loss))
        
        return {
            'loss': np.mean(epoch_loss),
            'cos_sim': np.mean(epoch_cos)
        }
    
    def run_validation_epoch(self, student_model, teacher_model, X, y):
        "Run validation"
        val_loss=[]
        val_cos=[]
        batch=32
        
        for i in range(0, len(X), batch):
            batch_X_t=X[i:i+batch]
            batch_y=y[i:i+batch]
            batch_X_s=batch_X_t.reshape(batch_X_t.shape[0], -1)
            
            teacher_layer_obj=teacher_model.layers[-1]
            teacher_strides=teacher_layer_obj.strides if isinstance(teacher_layer_obj, tf.keras.layers.Conv2D) else (1,1)
            teacher_features=teacher_model(batch_X_t, training=False)
            student_features=student_model(batch_X_s, training=False)
            teacher_flat=tf.reshape(teacher_features, [batch_X_t.shape[0], -1])
            student_flat=tf.reshape(student_features, [batch_X_s.shape[0], -1])
            total_loss=tf.reduce_mean(combined_similarity_loss(teacher_flat, student_flat))
            cos_val=1-total_loss
            val_loss.append(float(total_loss))
            val_cos.append(float(cos_val))
        return {
            'loss': np.mean(val_loss),
            'cos_sim': np.mean(val_cos)
        }
        
    def plot_training_curves(self, history, layer_idx):
        "Plot training and validation metrics" 
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        #Loss
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f"Layer {layer_idx} - Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        #Cosine Similarity
        plt.subplot(1, 2, 2)
        plt.plot(history['cos_sim'], label='Training Cosine')
        plt.plot(history['val_cos_sim'], label='Validation Cosine')
        plt.title(f"Layer {layer_idx} - Cosine Similarity")
        plt.xlabel('Epoch')
        plt.ylabel('Cosine Similarity')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def evaluate_layer(self, student_model, teacher_model, X, batch_size):
        "Evaluate layer performance on val dataset"
        losses=[]
        cosines=[]
        for i in range(0, len(X), batch_size):
            batch_X=X[i:i+batch_size]
            batch_y=teacher_model.predict(batch_X, verbose=0)
            if len(batch_y.shape)>2:
                batch_y=batch_y.reshape(batch_y.shape[0], -1)
            metrics=student_model.evaluate(batch_X, batch_y, verbose=0)
            losses.append(metrics[0])
            cosines.append(metrics[1])
        return np.mean(losses), np.mean(cosines)
    
    """def conv2d_to_dense_weights(self, conv_layer, input_shape):
        "Conv2D layer weights to Dense weights"
        print(f"\nConverting {conv_layer.name} to Dense layer")
        weights, biases=conv_layer.get_weights()
        strides=conv_layer.strides
        kernel_h, kernel_w, in_channels, out_channels=weights.shape
        if len(input_shape)==4:
            _, input_h, input_w, input_c=input_shape
        else:
            input_h, input_w, input_c=input_shape
            
        if input_c != in_channels:
            raise ValueError(f"Input channels mismatch: Conv2D expects {in_channels}, obtained {input_c}")
        
        if conv_layer.padding.lower()=='same':
            output_h=math.ceil(input_h/strides[0])
            output_w=math.ceil(input_w/strides[1])
        else:
            output_h=math.ceil((input_h-kernel_h)/strides[0])+1
            output_w=math.ceil((input_w-kernel_w)/strides[1])+1
            
        print(f"    Conv params: kernel={kernel_h}x{kernel_w}, in_ch={in_channels}, out_ch={out_channels}")
        print(f"    Input: {input_h}x{input_w}x{input_c}")
        print(f"    Output: {output_h}x{output_w}x{out_channels}")
        
        input_size=input_h*input_w*input_c
        output_size=output_h*output_w*out_channels
        if biases.shape!=(out_channels,):
            raise ValueError(f"Unexpected bias shape: {biases.shape}, obtained ({out_channels},)")
        
        dense_biases=np.zeros(output_size, dtype=np.float32)
        for out_c in range(out_channels):
            for out_y in range(output_h):
                for out_x in range(output_w):
                    out_idx=(out_y*output_w*out_channels)+(out_x*out_channels)+out_c
                    dense_biases[out_idx]=biases[out_c]
        
        dense_weights=np.zeros((input_size, output_size), dtype=np.float32)
        for out_y in range(output_h):
            for out_x in range(output_w):
                for out_c in range(out_channels):
                    out_idx=(out_y*output_w*out_channels)+(out_x*out_channels)+out_c
                    start_y=out_y*strides[0]
                    start_x=out_x*strides[1]
                    
                    if conv_layer.padding.lower()=='same':
                        #pad_y=(kernel_h-1)//2
                        #pad_x=(kernel_w-1)//2
                        pad_y = (output_h * strides[0] - input_h + kernel_h - strides[0]) // 2
                        pad_x = (output_w * strides[1] - input_w + kernel_w - strides[1]) // 2
                        start_y-=pad_y
                        start_x-=pad_x
                        
                    for ky in range(kernel_h):
                        for kx in range(kernel_w):
                            for in_c in range(in_channels):
                                input_y=start_y+ky
                                input_x=start_x+kx
                                if(0<=input_y<input_h) and (0<=input_x<input_w):
                                    in_idx=(input_y*input_w*input_c)+(input_x*input_c)+in_c
                                    dense_weights[in_idx, out_idx]=weights[ky, kx, in_c, out_c]
        #dense_weights/=np.prod(kernel_h*kernel_w)
        print(f"    Created weights shape: {dense_weights.shape}")
        print(f"    Created biases shape: {biases.shape}")
        print(f"Final biases shape: {dense_biases.shape}")
        return dense_weights, dense_biases"""
    
    def conv2d_to_dense_weights(self, conv_layer, input_shape):
        """Convert Conv2D layer weights to equivalent Dense layer weights (vectorized)."""
        print(f"\nConverting {conv_layer.name} to Dense layer")

        weights, biases = conv_layer.get_weights()
        strides = conv_layer.strides
        kernel_h, kernel_w, in_ch, out_ch = weights.shape
        
        if len(input_shape) == 4:
            _, in_h, in_w, in_c = input_shape
        else:
            in_h, in_w, in_c = input_shape
        if in_c != in_ch:
            raise ValueError(f"Input channels mismatch: expected {in_ch}, got {in_c}")
        if conv_layer.padding.lower() == 'same':
            out_h = math.ceil(in_h / strides[0])
            out_w = math.ceil(in_w / strides[1])
            pad_y = (out_h * strides[0] - in_h + kernel_h - strides[0])
            pad_x = (out_w * strides[1] - in_w + kernel_w - strides[1])
            pad_top=pad_y//2
            pad_bottom=pad_y-pad_top
            pad_left=pad_x//2
            pad_right=pad_x-pad_left 
        else:
            out_h = math.ceil((in_h - kernel_h) / strides[0]) + 1
            out_w = math.ceil((in_w - kernel_w) / strides[1]) + 1
            pad_top = pad_bottom = pad_left = pad_right = 0
        print(f"    Conv params: kernel={kernel_h}x{kernel_w}, in_ch={in_ch}, out_ch={out_ch}")
        print(f"    Input: {in_h}x{in_w}x{in_c}")
        print(f"    Output: {out_h}x{out_w}x{out_ch}")

        input_size = in_h * in_w * in_c
        output_size = out_h * out_w * out_ch
        if biases.shape != (out_ch,):
            raise ValueError(f"Unexpected bias shape: {biases.shape}, expected ({out_ch},)")
        dense_biases = np.tile(biases, out_h * out_w)
        out_y_idx, out_x_idx = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing='ij')
        out_y_idx = out_y_idx.flatten()
        out_x_idx = out_x_idx.flatten()
        start_y = out_y_idx * strides[0] - pad_top
        start_x = out_x_idx * strides[1] - pad_left
        dense_weights = np.zeros((input_size, output_size), dtype=np.float32)

        for ky in range(kernel_h):
            for kx in range(kernel_w):
                in_y = start_y + ky
                in_x = start_x + kx
                valid_mask = (in_y >= 0) & (in_y < in_h) & (in_x >= 0) & (in_x < in_w)
                if not np.any(valid_mask):
                    continue
                in_flat = (in_y[valid_mask] * in_w + in_x[valid_mask]) * in_c
                out_flat_base = np.where(valid_mask)[0] * out_ch
                for ic in range(in_ch):
                    in_idx = in_flat + ic
                    for oc in range(out_ch):
                        out_idx = out_flat_base + oc
                        dense_weights[in_idx, out_idx] = weights[ky, kx, ic, oc]

        print(f"    Created weights shape: {dense_weights.shape}")
        print(f"    Created biases shape: {dense_biases.shape}")
        return dense_weights, dense_biases
    
        
    def training(self, X_train, y_train, X_val, y_val, alpha=0.5, epochs_per_layer=30, final_epochs=300):
        "Training Phase per each Layer"
        print("\n"+"="*80)
        print("STARTING TRAINING PROCESS (PER-LAYER)")
        print("="*80)
        
        print("Training configuration:")
        print(f"    Feature weight: {alpha}")
        print(f"    Epochs per Layer (when trained): {epochs_per_layer}")
        print(f"    Training samples: {len(X_train)}")
        print(f"    Validation samples: {len(X_val)}")
        
        self.student_model=self.create_dense_student()
        self.set_mathematical_weights(self.student_model)
        layer_histories={}
        last_teacher_indices=sorted([idx for idx in self.layer_mapping.keys() if self.layer_mapping!=-1][-3:])
        last_student_indices=[self.layer_mapping[idx] for idx in last_teacher_indices]
        print(f"\nTraining {len(self.layer_mapping)} layers sequentially")
        for teacher_idx, student_idx in self.layer_mapping.items():
            if student_idx not in last_student_indices and student_idx!=-1:
                try:
                    history=self.train_layer_sequential(
                        teacher_idx=teacher_idx,
                        student_idx=student_idx,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        epochs=epochs_per_layer
                    )
                    layer_histories[f"layer_{student_idx}"]=history
                except Exception as e:
                    print(f"Error training layer {student_idx}: {e}")
                    continue
        
        print(f"\n"+"="*80)
        print("FINAL END-TO-END KD TRAINING")
        print("="*80)
        
        self.freeze_all_layers()
        for idx in last_student_indices: #enumerate(self.student_model.layers):#
            self.unfreeze_layer(idx)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        self.student_model.summary()
        #try:
        #    penultimate_layer_output=self.teacher_model.layers[-2].output
        #    output_layer_no_activation=Dense(units=self.num_classes, activation="relu", name='output_no_activation', trainable=False)(penultimate_layer_output)
        #    teacher_final_output_model=Model(
        #        inputs=self.teacher_model.input,
        #        outputs=output_layer_no_activation
        #    ) 
        #except IndexError:
        #    print("Warning: Teacher model has fewer than 2 layers. Using the last layer's output directly for distillation.")
        #    teacher_final_output_model = Model(
        #        inputs=self.teacher_model.input,
        #        outputs=self.teacher_model.layers[-1].output
        #    )
        
        #teacher_final_output_model.summary()
        #teacher_final_output_model.get_layer('output_no_activation').set_weights(self.teacher_model.layers[-1].get_weights())
        y_train_one=tf.keras.utils.to_categorical(y_train, num_classes=self.num_classes)
        y_val_one=tf.keras.utils.to_categorical(y_val, num_classes=self.num_classes)
        X_train=np.expand_dims(X_train, 1)
        X_train=np.transpose(X_train, (0, 2, 3, 1))
        X_val=np.expand_dims(X_val, 1) 
        X_val=np.transpose(X_val, (0, 2, 3, 1))
        X_train=X_train.reshape(X_train.shape[0], -1)
        X_val=X_val.reshape(X_val.shape[0], -1)
        
        print(f"Teacher model input shape: {teacher_dvec_model.input_shape}")
        print(f"X_train shape: {X_train.shape}")
        print(f"shape_input: {shape_input}")
        
        teacher_train_targets=teacher_dvec_model.predict(X_train)
        teacher_val_targets=teacher_dvec_model.predict(X_val)

        self.student_model.compile(
            optimizer=optimizer,
            loss=lambda y_true, y_pred: distillation_loss(
                y_true, y_pred, teacher_train_targets, alpha
            ),
            metrics=['accuracy', tf.keras.metrics.CategoricalAccuracy()]
        )
        
        history=self.student_model.fit(
            X_train,
            {'output': y_train, 'teacher': teacher_train_targets},
            batch_size=batch_size
        )
         
        train_loss_metric=tf.keras.metrics.Mean(name='train_loss')
        val_loss_metric=tf.keras.metrics.Mean(name='val_loss')
        train_acc_metric=tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        val_acc_metric=tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')
        
        def train_step_eager(x, y_true, teacher_targets):
            with tf.GradientTape() as tape:
                x=x.reshape(x.shape[0], -1)
                student_logits=self.student_model(x, training=True)
                per_sample_loss=distillation_loss(
                    y_true=y_true,
                    y_pred=student_logits,
                    teacher_targets=teacher_targets,
                    alpha=alpha
                )
                loss_value=tf.reduce_mean(per_sample_loss)
            trainable_vars=self.student_model.trainable_variables
            grads=tape.gradient(loss_value, trainable_vars)
            if grads is None:
                print("Warning: No gradients computed for trainable variables")
                return loss_value
            grads, _=tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(zip(grads, trainable_vars))
            train_loss_metric.update_state(loss_value)
            train_acc_metric.update_state(y_true, tf.nn.softmax(student_logits))
            return loss_value
    
        def test_step_eager(x, y_true, teacher_targets):
            x=x.reshape(x.shape[0], -1)
            student_logits=self.student_model(x, training=False)
            per_sample_loss=distillation_loss(
                y_true=y_true,
                y_pred=student_logits,
                teacher_targets=teacher_targets,
                alpha=alpha
            )
            loss_value=tf.reduce_mean(per_sample_loss)
            val_loss_metric.update_state(loss_value)
            val_acc_metric.update_state(y_true, tf.nn.softmax(student_logits))
            return loss_value 
        
        print(f"Final training with distillation loop for {final_epochs} epochs")
        batch_size=32
        
        for epoch in range(final_epochs):
            train_loss_metric.reset_state()
            val_loss_metric.reset_state()
            train_acc_metric.reset_state()
            val_acc_metric.reset_state()
            #train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
            #val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
            #train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
            #val_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')
            
            for i in range(0, X_train.shape[0], batch_size):
                batch_x=X_train[i:i+batch_size]
                batch_y=y_train_one[i:i+batch_size]
                batch_targets=teacher_train_targets[i:i+batch_size]
                _=train_step_eager(batch_x, batch_y, batch_targets)
                
            for i in range(0, X_val.shape[0], batch_size):
                batch_x_val=X_val[i:i+batch_size]
                batch_y_val=y_val_one[i:i+batch_size]
                batch_targets_val=teacher_val_targets[i:i+batch_size]
                _=test_step_eager(batch_x_val, batch_y_val, batch_targets_val)

            print(f"Epoch {epoch+1}/{final_epochs} - Loss: {train_loss_metric.result():.4f}, Accuracy: {train_acc_metric.result():.4f}, Val Loss: {val_loss_metric.result():.4f}, Val Accuracy: {val_acc_metric.result():.4f}")
        
        print("Final block training completed successfully!")
        self.save_student_model('distilled_student_model.h5')
        return self.student_model, {'layer_histories': layer_histories}
    
    def save_student_model(self, filepath='student_model.h5'):
        "Save the student model to a file"
        if self.student_model is not None:
            self.student_model.save(filepath)
            print(f"Student model saved successfully to {filepath}")
        else:
            print("Warning: No student model exists to save")


fc_onetoone_model=CompleteTransitionToFC(teacher_dvec_model, shape_input)
student_model, history=fc_onetoone_model.training(
    X_train, y_train, X_val, y_val,
    alpha=0.5,
    epochs_per_layer=20,
    final_epochs=300
)        