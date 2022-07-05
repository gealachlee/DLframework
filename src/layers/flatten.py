from layers.base_layer import LayerType,Layer

class FlattenLayer(Layer):
    layerType = LayerType.FlattenLayer
    def __init__(self):
        self.units = 0

    def set_units(self,units):
        self.units=units

    def forward(self, node_in,*args, **kwargs):
        self.h=node_in.reshape(node_in.shape[0],-1)#保持N
        return self.h

    def backward(self, h_1, delta_plus_1, ind, **kwargs):
        self.dz = delta_plus_1

    def update(self,*args,**kwargs):
        pass

    def set_initialized(self, _is_initialized: bool):
        pass

    def get_units(self):
        return 0
