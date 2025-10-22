# Energy-Efficient Accelerator Architecture for Transformers Using Linear Attention

This is the repo for ECE562 project **Energy-Efficient Accelerator Architecture for Transformers Using Linear Attention**.

## Structure

```
    /models
        models after training and quantization
        /xxx/checkpoint.pth
            float model after training
        /xxx/model_float.onnx
            float ONNX model
        /xxx/model_quantized.onnx
            ONNX model with linear and matmul nodes quantized
    /ViTALiTy
        modified code, support degree-1 and degree-2 taylor expression of attention
```


## Acknowledge

The code refers to:

https://github.com/GATECH-EIC/ViTALiTy

https://github.com/AMD-AGI/AMD_QTViT
