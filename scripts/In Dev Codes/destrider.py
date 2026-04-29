import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
import numpy as np

def create_compatible_teacher_model(teacher_model):
    """
    Improved version with better weight transfer and layer handling
    """
    input_tensor = Input(shape=teacher_model.input_shape[1:])
    x = input_tensor
    layer_mapping = {}  # Map original layer names to new layers
    
    print("Processing layers...")
    
    for i, layer in enumerate(teacher_model.layers[1:]):  # Skip input layer
        print(f"Processing layer {i+1}: {layer.name} ({type(layer).__name__})")
        
        if isinstance(layer, Conv2D) and (layer.strides[0] > 1 or layer.strides[1] > 1):
            # Replace strided conv with regular conv + pooling
            print(f"  Converting strided conv (stride={layer.strides}) to conv+pool")
            print(f"    Input shape: {x.shape}")
            
            # Create new conv layer with stride=1
            new_conv = Conv2D(
                filters=layer.filters,
                kernel_size=layer.kernel_size,
                strides=(1, 1),  # Always use stride 1
                padding=layer.padding,
                activation=layer.activation,
                use_bias=layer.use_bias,
                name=f"{layer.name}_conv1x1"
            )(x)
            
            # Calculate appropriate pool size based on input dimensions and desired output
            # We want to match the spatial reduction of the original strided conv
            input_h, input_w = x.shape[1], x.shape[2]
            stride_h, stride_w = layer.strides
            
            # For small spatial dimensions, use smaller pool sizes to avoid over-reduction
            if input_h <= 4 or input_w <= 4:
                # Use stride of 1 for very small inputs, or minimal pooling
                pool_size = min(2, stride_h)
                pool_stride = min(2, stride_h)
                padding_mode = 'same'  # Preserve some spatial info
            else:
                pool_size = stride_h
                pool_stride = stride_h  
                padding_mode = 'valid'
            
            print(f"    Using pool_size={pool_size}, stride={pool_stride}, padding={padding_mode}")
            
            new_pool = MaxPooling2D(
                pool_size=(pool_size, pool_size),
                strides=(pool_stride, pool_stride),
                padding=padding_mode,
                name=f"{layer.name}_pool"
            )(new_conv)
            
            print(f"    Output shape after conv+pool: {new_pool.shape}")
            x = new_pool
            layer_mapping[layer.name] = (new_conv, new_pool)
            
        else:
            # Handle other layer types
            if isinstance(layer, BatchNormalization):
                new_layer = BatchNormalization(
                    axis=layer.axis,
                    momentum=layer.momentum,
                    epsilon=layer.epsilon,
                    center=layer.center,
                    scale=layer.scale,
                    name=f"{layer.name}_new"
                )(x)
                
            elif isinstance(layer, Conv2D):
                new_layer = Conv2D(
                    filters=layer.filters,
                    kernel_size=layer.kernel_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    activation=layer.activation,
                    use_bias=layer.use_bias,
                    name=f"{layer.name}_new"
                )(x)
                
            elif isinstance(layer, Dense):
                # Flatten if needed before dense layer
                if len(x.shape) > 2:
                    x = Flatten(name=f"flatten_before_{layer.name}")(x)
                
                new_layer = Dense(
                    units=layer.units,
                    activation=layer.activation,
                    use_bias=layer.use_bias,
                    name=f"{layer.name}_new"
                )(x)
                
            elif isinstance(layer, Dropout):
                new_layer = Dropout(
                    rate=layer.rate,
                    name=f"{layer.name}_new"
                )(x)
                
            elif isinstance(layer, MaxPooling2D):
                new_layer = MaxPooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=f"{layer.name}_new"
                )(x)
                
            elif isinstance(layer, Flatten):
                new_layer = Flatten(name=f"{layer.name}_new")(x)
                
            else:
                print(f"  Warning: Unhandled layer type {type(layer).__name__}")
                # Try to clone the layer
                try:
                    config = layer.get_config()
                    if 'batch_input_shape' in config:
                        del config['batch_input_shape']
                    config['name'] = f"{layer.name}_new"
                    new_layer = layer.__class__.from_config(config)(x)
                except Exception as e:
                    print(f"  Error cloning layer: {e}")
                    continue
            
            x = new_layer
            layer_mapping[layer.name] = new_layer
    
    # Create new model
    new_model = Model(inputs=input_tensor, outputs=x, name="modified_model")
    
    # Transfer weights with improved logic
    print("\nTransferring weights...")
    
    # Build a mapping from original layer names to new model layer names
    name_mapping = {}
    for orig_layer in teacher_model.layers[1:]:  # Skip input layer
        if orig_layer.name in layer_mapping:
            mapped_item = layer_mapping[orig_layer.name]
            
            if isinstance(mapped_item, tuple):
                # This is a strided conv that was split into conv + pool
                conv_tensor, pool_tensor = mapped_item
                # Find the actual layer in the new model by looking for the conv layer
                for new_layer in new_model.layers:
                    if new_layer.name == f"{orig_layer.name}_conv1x1":
                        name_mapping[orig_layer.name] = new_layer.name
                        break
            else:
                # Regular layer mapping - find by the new name pattern
                for new_layer in new_model.layers:
                    if new_layer.name == f"{orig_layer.name}_new":
                        name_mapping[orig_layer.name] = new_layer.name
                        break
    
    # Now transfer weights using the actual layer objects from the built model
    for orig_layer in teacher_model.layers[1:]:  # Skip input layer
        if orig_layer.name in name_mapping:
            new_layer_name = name_mapping[orig_layer.name]
            
            try:
                new_layer = new_model.get_layer(new_layer_name)
                
                if orig_layer.weights and new_layer.weights:
                    orig_weights = orig_layer.get_weights()
                    
                    # Check if weight shapes match
                    if len(orig_weights) == len(new_layer.weights):
                        shape_match = all(
                            orig_w.shape == new_w.shape 
                            for orig_w, new_w in zip(orig_weights, new_layer.weights)
                        )
                        
                        if shape_match:
                            new_layer.set_weights(orig_weights)
                            print(f"  ✓ Transferred weights: {orig_layer.name} -> {new_layer_name}")
                        else:
                            print(f"  ✗ Weight shape mismatch for {orig_layer.name}")
                            for i, (orig_w, new_w) in enumerate(zip(orig_weights, new_layer.weights)):
                                print(f"    Weight {i}: {orig_w.shape} vs {new_w.shape}")
                    else:
                        print(f"  ✗ Different number of weights for {orig_layer.name}: {len(orig_weights)} vs {len(new_layer.weights)}")
                        
            except Exception as e:
                print(f"  ✗ Failed to transfer weights for {orig_layer.name}: {e}")
    
    return new_model

def diagnose_model_issues(original_model, modified_model, test_input):
    """
    Diagnose potential issues with the modified model
    """
    print("\n=== Model Diagnostics ===")
    
    # Get intermediate outputs to see where things diverge
    print("Checking intermediate layer outputs...")
    
    # Create models that output intermediate layers
    orig_layers_of_interest = ['conv2d_4', 'conv2d_5', 'conv2d_6', 'conv2d_7', 'fully_connected']
    modified_layers_of_interest = ['conv2d_4_new', 'conv2d_5_new', 'conv2d_6_conv1x1', 'conv2d_7_conv1x1', 'fully_connected_new']
    
    for orig_name, mod_name in zip(orig_layers_of_interest, modified_layers_of_interest):
        try:
            # Get outputs from both models
            orig_layer_model = Model(inputs=original_model.input, outputs=original_model.get_layer(orig_name).output)
            mod_layer_model = Model(inputs=modified_model.input, outputs=modified_model.get_layer(mod_name).output)
            
            orig_out = orig_layer_model.predict(test_input, verbose=0)
            mod_out = mod_layer_model.predict(test_input, verbose=0)
            
            # Calculate statistics
            orig_stats = f"mean={np.mean(orig_out):.4f}, std={np.std(orig_out):.4f}, min={np.min(orig_out):.4f}, max={np.max(orig_out):.4f}"
            mod_stats = f"mean={np.mean(mod_out):.4f}, std={np.std(mod_out):.4f}, min={np.min(mod_out):.4f}, max={np.max(mod_out):.4f}"
            
            print(f"{orig_name:15} | {orig_stats}")
            print(f"{mod_name:15} | {mod_stats}")
            print(f"{'Difference':15} | mean_diff={np.mean(np.abs(orig_out - mod_out)):.4f}")
            print()
            
        except Exception as e:
            print(f"Could not compare {orig_name} vs {mod_name}: {e}")
    
    # Check final outputs in detail
    orig_final = original_model.predict(test_input, verbose=0)
    mod_final = modified_model.predict(test_input, verbose=0)
    
    print("Final output analysis:")
    print(f"Original top-5: {np.argsort(orig_final[0])[-5:][::-1]} with probs: {np.sort(orig_final[0])[-5:][::-1]}")
    print(f"Modified top-5: {np.argsort(mod_final[0])[-5:][::-1]} with probs: {np.sort(mod_final[0])[-5:][::-1]}")
    
    # Check for saturation
    if np.max(mod_final) > 0.99 and len(np.where(mod_final[0] > 0.01)[0]) < 5:
        print("⚠️  Modified model shows signs of saturation (one very high probability)")
    
    # Check for NaN or extreme values
    if np.any(np.isnan(mod_final)) or np.any(np.isinf(mod_final)):
        print("⚠️  Modified model has NaN or Inf values!")
    
    if np.max(np.abs(mod_final)) > 100:
        print("⚠️  Modified model has extreme output values!")

def compare_models(original_model, modified_model, num_samples=5):
    """
    Compare outputs of original and modified models with diagnostics
    """
    print(f"\n=== Model Comparison with {num_samples} samples ===")
    
    input_shape = original_model.input_shape[1:]
    total_agreement = 0
    total_diff = 0
    
    # Generate a fixed test input for diagnostics
    fixed_test_input = np.random.seed(42)  # For reproducibility
    diagnostic_input = np.random.rand(1, *input_shape)
    
    # Run diagnostics on the first sample
    if num_samples > 0:
        diagnose_model_issues(original_model, modified_model, diagnostic_input)
    
    for i in range(num_samples):
        # Generate random test input
        test_input = np.random.rand(1, *input_shape)
        
        # Get predictions
        orig_output = original_model.predict(test_input, verbose=0)
        new_output = modified_model.predict(test_input, verbose=0)
        
        # Calculate metrics
        output_diff = np.mean(np.abs(orig_output - new_output))
        
        orig_pred = np.argmax(orig_output, axis=-1)[0]
        new_pred = np.argmax(new_output, axis=-1)[0]
        agreement = int(orig_pred == new_pred)
        
        total_agreement += agreement
        total_diff += output_diff
        
        # Show more detailed output for first few samples
        if i < 3:
            print(f"Sample {i+1}:")
            print(f"  Original prediction: {orig_pred} (confidence: {np.max(orig_output):.4f})")
            print(f"  Modified prediction: {new_pred} (confidence: {np.max(new_output):.4f})")
            print(f"  Output difference: {output_diff:.6f}")
            print(f"  Agreement: {'✓' if agreement else '✗'}")
            print()
    
    avg_agreement = (total_agreement / num_samples) * 100
    avg_diff = total_diff / num_samples
    
    print(f"Summary:")
    print(f"  Average output difference: {avg_diff:.6f}")
    print(f"  Average class agreement: {avg_agreement:.1f}%")
    
    return avg_agreement, avg_diff

# Load and convert model
print("Loading original model...")
teacher_model = tf.keras.models.load_model("cnn-librispeech-classifier.h5")

print("\nOriginal model summary:")
teacher_model.summary()

print("\nConverting model...")
modified_model = create_compatible_teacher_model(teacher_model)

print("\nModified model summary:")
modified_model.summary()

# Compare models
agreement, diff = compare_models(teacher_model, modified_model, num_samples=10)

# Save the modified model
print("\nSaving modified model...")
modified_model.save("improved_modified_model.keras")  # Use .keras format
print("Model saved as 'improved_modified_model.keras'")

# Additional analysis
print(f"\n=== Model Analysis ===")
print(f"Original model parameters: {teacher_model.count_params():,}")
print(f"Modified model parameters: {modified_model.count_params():,}")

if agreement < 50:
    print(f"\n⚠️  Low agreement ({agreement:.1f}%) suggests significant architectural differences.")
    print("Consider:")
    print("1. Using smaller pool sizes (e.g., 2x2 instead of matching original stride)")
    print("2. Adding padding to maintain spatial dimensions")
    print("3. Fine-tuning the modified model on your data")
elif agreement < 90:
    print(f"\n⚠️  Moderate agreement ({agreement:.1f}%). Some differences expected due to architectural changes.")
else:
    print(f"\n✅ Good agreement ({agreement:.1f}%)!")