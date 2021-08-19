import tensorflow as tf
import numpy

class Conv2DLowRank(tf.keras.layers.Layer):
    def __init__(self, filters, rank, kernel_size, strides=(1,1), 
                 padding='valid', activation=None, use_bias=True, **kwargs):
        super(Conv2DLowRank, self).__init__(**kwargs)
        self.filters = filters
        self.rank = rank
        self.kernel_size = 2*(kernel_size,) if isinstance(kernel_size, int) else kernel_size
        self.strides = 2*(strides,) if isinstance(strides, int) else strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        v, h = self.kernel_size
        vs, hs = self.strides
        self.v_layer = tf.keras.layers.Conv2D(rank, kernel_size=(v,1), strides=(vs,1), 
                                              padding=padding, activation='linear', use_bias=False)
        self.h_layer = tf.keras.layers.Conv2D(filters, kernel_size=(1,h), strides=(1,hs), 
                                              padding=padding, activation=activation, use_bias=use_bias)

    def call(self, inputs):
        return self.h_layer(self.v_layer(inputs))
    
    def get_config(self):
        config = super(Conv2DLowRank, self).get_config()
        config.update({'filters':self.filters, 'rank':self.rank, 'kernel_size':self.kernel_size, 
                       'strides':self.strides, 'padding':self.padding, 'activation':self.activation, 
                       'use_bias':self.use_bias})
        return config


def LR_clone(layer, weight_dict, ratio=0.95): # ratio for automatic rank choosing
    assert ratio == 'manual' or ratio < 1
    if isinstance(layer, tf.keras.layers.Conv2D):
        w = layer.weights[0].numpy()
        w = w.transpose((2, 0, 1, 3))
        C = w.shape[0]
        N = w.shape[3]
        d = w.shape[1]
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D): # or 1 in w.shape
            return layer.__class__.from_config(layer.get_config())
        assert w.shape[1] == w.shape[2]
        shape2d = (w.shape[0]*w.shape[1], w.shape[2]*w.shape[3]) # (Cd x dN)
        w2d = numpy.empty(shape2d, dtype='float32')
        for i1 in range(C):
            for i2 in range(d):
                for i3 in range(d):
                    for i4 in range(N):
                        w2d[i1*d+i2, i4*d+i3] = w[i1,i2,i3,i4]
        U, D, Q = numpy.linalg.svd(w2d)
        Q = Q.transpose()
        
        if ratio == 'manual':
            K = weight_dict[layer.name]
        else:
            minVar = ratio * D.sum() # 0.95 default
            K = 1 
            for i in D.cumsum(): 
                if i > minVar:
                    break
                K += 1        
        if K >= d*C*N/(C+N): # K < dCN/(C+N) to accelerate
            return layer.__class__.from_config(layer.get_config())
        
        D = numpy.sqrt(D)          
        V = numpy.empty((d,1,C,K), dtype='float32')
        H = numpy.empty((1,d,K,N), dtype='float32')
        for k in range(K):
            for c in range(C):
                V[:, 0, c, k] = U[c*d : c*d+d, k] * D[k]
        for n in range(N):
            for k in range(K):
                H[0, :, k, n] = Q[n*d : n*d+d, k] * D[k]        
                
        config = layer.get_config()
        name = config['name']
        stride = config['strides']
        pad = config['padding']
        act = config['activation']
        bias = config['use_bias']
        newlayer = Conv2DLowRank(N, K, d, strides=stride, padding=pad, 
                                 activation=act, use_bias=bias, name=name)
        weight_dict[newlayer.name] = [V, H] # weights are stored to assign them later
        if bias: weight_dict[newlayer.name].append(layer.weights[1].numpy())
        return newlayer
    else:
        return layer.__class__.from_config(layer.get_config())

def LR_optimize(model, ratio=0.95, LR_weights=dict()):
    clone = tf.keras.models.clone_model(model, clone_function=lambda l: LR_clone(l, LR_weights, ratio))
    for i in range(len(clone.layers)):
        layer = model.get_layer(index=i)
        clone_layer = clone.get_layer(index=i)
        if isinstance(clone_layer, Conv2DLowRank):
            weights = LR_weights[clone_layer.name] # [V,H,b]
            clone_layer.v_layer.set_weights(weights[:1])
            clone_layer.h_layer.set_weights(weights[1:])
        else:
            clone_layer.set_weights(layer.get_weights())

    return clone
