class Conv2DDecomp(tf.keras.layers.Layer):
    def __init__(self, filters, rank, kernel_size, strides=(1,1), 
                 padding='valid', activation=None, use_bias=True, **kwargs):
        super(Conv2DDecomp, self).__init__(**kwargs)
        self.filters = filters
        self.rank = rank
        self.kernel_size = 2*(kernel_size,) if isinstance(kernel_size, int) else kernel_size
        self.strides = 2*(strides,) if isinstance(strides, int) else strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.in_layer = tf.keras.layers.Conv2D(rank, kernel_size=1, activation=None, 
                                              use_bias=False)
        if self.kernel_size != (1,1) or self.strides != (1,1):
            self.conv_layer = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, 
                                                           padding=padding, activation=None, use_bias=False)
        else:
            self.conv_layer = None
        self.out_layer = tf.keras.layers.Conv2D(filters, kernel_size=1, activation=activation, 
                                              use_bias=use_bias)

    def call(self, inputs):
        if self.conv_layer:
            return self.out_layer(self.conv_layer(self.in_layer(inputs)))
        else:
            return self.out_layer(self.in_layer(inputs))
    
    def get_config(self):
        config = super(Conv2DDecomp, self).get_config()
        config.update({'filters':self.filters, 'rank':self.rank, 'kernel_size':self.kernel_size, 
                       'strides':self.strides, 'padding':self.padding, 'activation':self.activation, 
                       'use_bias':self.use_bias})
        return config
    
    
class DenseDecomp(tf.keras.layers.Layer):
    def __init__(self, units, rank, activation=None, use_bias=True, **kwargs):
        super(DenseDecomp, self).__init__(**kwargs)
        self.units = units
        self.rank = rank
        self.activation = activation
        self.use_bias = use_bias
        self.in_layer = tf.keras.layers.Dense(rank, activation=None, use_bias=False)
        self.out_layer = tf.keras.layers.Dense(units, activation=activation, use_bias=use_bias)

    def call(self, inputs):
        return self.out_layer(self.in_layer(inputs))
    
    def get_config(self):
        config = super(DenseDecomp, self).get_config()
        config.update({'units':self.units, 'rank':self.rank, 
                       'activation':self.activation, 
                       'use_bias':self.use_bias})
        return config
    

def norm_dist(a, b):
    return np.sum((a - b) ** 2) ** 0.5

# Keras dxdxCxN -> dxdxR CxR RxN

def CP_decompose(weight_tensor, rank): # use TPM to decompose 4d tensor
    X = weight_tensor.copy()
    U = np.empty(X.shape[:2] + (rank,), 'float32')
    V = np.empty((X.shape[2], rank), 'float32')
    W = np.empty((X.shape[3], rank), 'float32')
    for i in range(rank):
        # initialize normalized tensors
        Ui = np.random.randn(np.prod(U.shape[:2])).reshape(U.shape[:2])
        Ui /= np.sum(Ui**2) ** 0.5
        Vi = np.random.randn(V.shape[0])
        Vi /= np.sum(Vi**2) ** 0.5
        Wi = np.random.randn(W.shape[0])
        Wi /= np.sum(Wi**2) ** 0.5
        while True:
            Up, Vp, Wp = Ui.copy(), Vi.copy(), Wi.copy()
            Ui = np.tensordot(np.tensordot(X, Vi, axes=(2,0)), Wi, axes=(2,0)) # X x2 Vi x3 Wi
            Ui /= np.sum(Ui**2) ** 0.5
            Vi = np.tensordot(np.tensordot(X, Ui, axes=((0,1),(0,1))), Wi, axes=(1,0)) # X x1 Ui x3 Wi
            Vi /= np.sum(Vi**2) ** 0.5
            Wi = np.tensordot(np.tensordot(X, Ui, axes=((0,1),(0,1))), Vi, axes=(0,0)) # X x1 Ui x2 Vi
            Wi /= np.sum(Wi**2) ** 0.5
            if norm_dist(Ui,Up) < 1e-3 and norm_dist(Vi,Vp) < 1e-3 and norm_dist(Wi,Wp) < 1e-3:
                break # stop when convergence is reached
        d = np.dot(np.tensordot(np.tensordot(X, Ui, axes=((0,1),(0,1))), Vi, axes=(0,0)), Wi) # d 
        U[...,i] = Ui
        # spread the d factor
        V[:,i] = Vi * d ** 0.5
        W[:,i] = Wi * d ** 0.5
        X -= np.tensordot(np.tensordot(Ui, Vi, axes=0), Wi, axes=0) 
    
    return U.reshape(U.shape+(1,)), V.reshape((1,1)+V.shape), W.T.reshape((1,1)+W.T.shape)

def SVD_decompose(weight_tensor, rank): # use SVD to decompose a 2nd order tensor
    conv = len(weight_tensor.shape) == 4
    weight_tensor = weight_tensor.squeeze()
    U, D, Q = np.linalg.svd(weight_tensor, full_matrices=False)
    D = np.sqrt(D[:rank])          
    U = U[:,:rank] * D
    Q = np.transpose(Q[:rank,:].T * D)
    if conv:
        return U.reshape((1,1)+U.shape), Q.reshape((1,1)+Q.shape)
    else:
        return U, Q


def Decomp_clone(layer, ranks, fullcopy=False):
    if layer.name in ranks and not isinstance(layer, (Conv2DDecomp, DenseDecomp)):
        config = layer.get_config()
        if isinstance(layer, tf.keras.layers.Conv2D):
            w_shape = layer.weights[0].shape
            name = config['name']
            kernel = config['kernel_size']
            stride = config['strides']
            pad = config['padding']
            act = config['activation']
            bias = config['use_bias']
            return Conv2DDecomp(w_shape[-1], ranks[layer.name], kernel_size=kernel, strides=stride, 
                                padding=pad, activation=act, use_bias=bias, name=name)
        elif isinstance(layer, tf.keras.layers.Dense):
            w_shape = layer.weights[0].shape
            name = config['name']
            act = config['activation']
            bias = config['use_bias']
            return DenseDecomp(w_shape[-1], ranks[layer.name], activation=act, use_bias=bias, name=name)
    elif fullcopy:
        return layer.__class__.from_config(layer.get_config())
    else:
        return layer

def CP_optimize(model, layer_ranks=5, fullcopy=False):
    ranks = layer_ranks if isinstance(layer_ranks, dict) else dict()
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                pass
            elif isinstance(layer_ranks, int):
                ranks[layer.name] = layer_ranks
    clone = tf.keras.models.clone_model(model, clone_function=lambda l: Decomp_clone(l, ranks, fullcopy))
    
    for i in range(len(clone.layers)):
        layer = model.get_layer(index=i)
        clone_layer = clone.get_layer(index=i)
        if layer.name in ranks and not isinstance(layer, (Conv2DDecomp, DenseDecomp)):
            if hasattr(clone_layer, 'conv_layer') and clone_layer.conv_layer:
                U, V, W = CP_decompose(layer.weights[0].numpy(), ranks[layer.name])
                clone_layer.conv_layer.set_weights([U])
            else:
                V, W = SVD_decompose(layer.weights[0].numpy(), ranks[layer.name])
            clone_layer.in_layer.set_weights([V])
            if layer.get_config()['use_bias']:
                b = layer.weights[1].numpy()
                clone_layer.out_layer.set_weights([W,b])
            else:
                clone_layer.out_layer.set_weights([W])
        else:
            clone_layer.set_weights(layer.get_weights())

    return clone

