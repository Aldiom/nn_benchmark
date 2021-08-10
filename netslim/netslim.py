import tensorflow as tf
import numpy as np

class ChannelSelectWrapper(tf.keras.layers.Layer):
    def __init__(self, layer, indices, channels_last=True, **kwargs):
        super(ChannelSelectWrapper, self).__init__(**kwargs)
        self.layer = layer
        self.indices = indices
        self.channels_last = channels_last

    def call(self, inputs):
        ax = -1 if self.channels_last else 1
        return self.layer(tf.gather(inputs, self.indices, axis=ax))
    
    def get_config(self):
        config = super(ChannelSelectWrapper, self).get_config()
        config.update({'layer':self.layer, 'indices':self.indices, 'channels_last': self.channels_last})
        return config


def prune_clone(layer, to_prune):
    config = layer.get_config() 
    try:
        if layer.input.name in to_prune and 'use_bias' in config:
            config['use_bias'] = to_prune[layer.input.name] # True
        if layer.output.name in to_prune and 'filters' in config:
            config['filters'] = to_prune[layer.output.name] # unprunned channels
    except AttributeError:
        pass
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        config['gamma_regularizer'] = None
    newlayer = layer.__class__.from_config(config)
    if layer.name in to_prune: # unprunned indices
        return ChannelSelectWrapper(newlayer, to_prune[layer.name], name=config['name']+'_W') 
    return newlayer


ef apply_pruning(model, fraction): # prune a given fraction of channels
    assert fraction < 1, 'fraction must be <1'
    bn_channels = []
    bn_idx = []
    num_channels = 0
    # collect all gammas
    for i in range(len(model.layers)):
        layer = model.layers[i]
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            weights = layer.gamma.numpy()
            num_channels += weights.size
            bn_channels.append(weights)
            bn_idx.append(i)
    gammas = np.empty(num_channels)
    idx = 0
    for l in bn_channels:
        length = l.size
        gammas[idx:idx+length] = l
        idx += length
    gammas.sort()
    cutoff = gammas[int(round(fraction * num_channels))] # get the cutoff threshold
    prune_ref = dict()
    in_connect = dict() # dict that says which BN layer a given tensor goes to, starts backwards
    out_connect = dict() # dict of sets of layers at the output of a given tensor
    unprunnable = set()
    for i in range(len(model.layers)): # determine which layers should be pruned
        if i in bn_idx: # if layer is BN
            bn_layer =  model.layers[i]
            bn_gammas = bn_layer.gamma.numpy()
            not_pruning = np.sum(bn_gammas > cutoff)
            if bn_gammas.size > not_pruning: # if there are prunned channels 
                prune_ref[bn_layer.input.name] = not_pruning # no. of channels that prev. conv layer must have
                in_connect[bn_layer.input.name] = bn_layer.name # to get which conv layer is at input
                out_connect[bn_layer.output.name] = set() # here go the layers connected to output
                
    changed = True # propagate foreprune search
    while changed: 
        changed = False
        for l in model.layers:
            if isinstance(l.input, list): # if add layer or similar
                for i in l.input:
                    if i.name in out_connect: # if BN at input
                        out_connect[i.name].add(l) 
                        unprunnable.add(i.name) # dont prune, still have to look for layer that goes to tensor
            elif l.input.name in out_connect: # if it is at the output of a BN
                out_connect[l.input.name].add(l)
                channel_passthru = type(l) in (tf.keras.layers.Activation, tf.keras.layers.ReLU,
                                               tf.keras.layers.MaxPool2D, tf.keras.layers.ZeroPadding2D, 
                                               tf.keras.layers.Dropout)# tf.keras.layers.DepthwiseConv2D)
                if channel_passthru and l.output.name not in out_connect:
                    out_connect[l.output.name] = set() # propagate
                    changed = True
                if type(l) in (tf.keras.layers.DepthwiseConv2D, 
                               tf.keras.layers.GlobalAveragePooling2D,
                               tf.keras.layers.Flatten):
                    unprunnable.add(l.input.name + 'HARD') # dont prune if goes to DW or global/flatten (hard)
                elif isinstance(l, tf.keras.layers.Conv2D):
                    prune_ref[l.input.name] = True 
                                
    changed = True
    while changed: # propagate backprune search
        changed = False
        for l in model.layers: 
            if l.output.name in in_connect: 
                if type(l) in (tf.keras.layers.Add,): 
                    unprunnable.add(in_connect[l.output.name]) # if BN has Add at input, dont prune
                    prune_ref.pop(l.output.name, None) # Adds can't be touched
                    continue
                elif isinstance(l, tf.keras.layers.DepthwiseConv2D) and l.output.name in prune_ref:
                    prune_ref[l.output.name] = l.name # write down name for later
                    unprunnable.add(in_connect[l.output.name] + 'DW') 
                elif not isinstance(l, tf.keras.layers.Conv2D):
                    prune_ref[l.input.name] = prune_ref.pop(l.output.name) # propagate search
                    in_connect[l.input.name] = in_connect.pop(l.output.name)
                    changed = True
            chk_unprunnable = l.output.name in unprunnable or l.output.name+'HARD' in unprunnable
            if chk_unprunnable: # means that its output goes to an Add (or DW)
                hardness = '' if l.output.name in unprunnable else 'HARD'
                if isinstance(l, tf.keras.layers.BatchNormalization):
                    unprunnable.add(l.name + hardness)
                elif not isinstance(l, tf.keras.layers.Add): # or others that take several inputs
                    unprunnable.add(l.input.name + hardness)
                    changed = True
                unprunnable.remove(l.output.name + hardness)

    for i in list(in_connect.keys()):
        in_connect[in_connect.pop(i)] = i   
                
    for l in list(unprunnable): # just BN names
        actually_dw = False
        if l[-4:] == 'HARD': # hard unprunnables are never modified
            l_name = l[:-4]
            hardness = True
            unprunnable.add(l_name)
            unprunnable.remove(l)
        else:
            l_name = l
            if l[-2:] == 'DW':
                l_name = l[:-2]
                unprunnable.remove(l)
                actually_dw = True # actually prunnable, but this works to deal with DWs too
            hardness = False 
        bn_layer = model.get_layer(l_name)
        bn_gammas = bn_layer.gamma.numpy()
        indices = bn_gammas > cutoff 
        if actually_dw: # it is assumed DW is just before BN
            prune_ref[prune_ref.pop(bn_layer.input.name)] = np.flatnonzero(indices)
            continue
        if not hardness:
            out_name = bn_layer.output.name
            for out in out_connect[out_name]: # wrap all layers at output that arent Adds
                if not isinstance(out, tf.keras.layers.Add) and indices.size > indices.sum():
                    prune_ref[out.name] = np.flatnonzero(indices)
        prune_ref.pop(in_connect[l_name], None) # conv before BN can't be pruned
            
    
    new_model = tf.keras.models.clone_model(model, clone_function=lambda l: prune_clone(l, prune_ref))
                 
    forward_prune = dict()
    backward_prune = dict()
    for i in range(len(new_model.layers)):
        oldlayer = model.layers[i] 
        newlayer = new_model.layers[i]
        if isinstance(newlayer, ChannelSelectWrapper): 
            newlayer = newlayer.layer # adjust reference to modify wrapped layer (not the wrap itself)
        
        if isinstance(oldlayer, tf.keras.layers.BatchNormalization):
            weights = oldlayer.get_weights()
            unprunned = weights[0] > cutoff
            if oldlayer.name in unprunnable: # if unprunnable do nothing
                newlayer.set_weights(weights)
            else: # else, copy unprunned channels params
                newlayer.set_weights([weights[n][unprunned] for n in range(4)])
            if unprunned.size == unprunned.sum(): # check if pruned or not, depends on gammas magnitudes
                continue 
            elif oldlayer.name not in unprunnable: # with this you later see which weights are copied 
                backward_prune[in_connect[oldlayer.name]] = unprunned # in conv before this BN
            outref = oldlayer.output.name
            activ = 'linear'
            changed = True 
            while changed: # if there's pruning, look for layers at output to adjust its params
                changed = False # search is propagated thru the net to find all connected convs
                for outlayer in out_connect[outref]: # TODO add more activation layers
                    if isinstance(outlayer, tf.keras.layers.ReLU):
                        activ = 'relu' + str(outlayer.get_config()['max_value'])
                        outref = outlayer.output.name
                        changed = True
                        break
                    elif isinstance(outlayer, tf.keras.layers.Activation):
                        activ = outlayer.get_config()['activation']
                        outref = outlayer.output.name
                        changed = True
                        break
                    elif type(outlayer) in (tf.keras.layers.ZeroPadding2D, 
                                            tf.keras.layers.MaxPooling2D): # maybe add dropout someday
                        outref = outlayer.output.name
                        changed = True
                        break
            betas = weights[1] 
            if 'relu' in activ: # save and propagate (thru act layers and such) betas of prunned channels
                saturate = None if activ=='relu' else float(activ[4:])
                betas = np.clip(betas, 0, saturate)
            elif activ == 'sigmoid':
                betas = 1 / (1 + np.exp(-betas))
            elif activ == 'softmax':
                betas = np.exp(betas) / np.sum(np.exp(betas)) 
            forward_prune[outref] = (unprunned, betas) # saved here de adjust input weights of layers 
                                             # after the output of this one
        elif not isinstance(oldlayer, tf.keras.layers.Conv2D):
            newlayer.set_weights(oldlayer.get_weights())
            
    for i in range(len(new_model.layers)): # 2nd pass to cover convs
        oldlayer = model.layers[i] 
        newlayer = new_model.layers[i]
        if isinstance(newlayer, ChannelSelectWrapper):
            newlayer = newlayer.layer # same as above
            
        if isinstance(oldlayer, tf.keras.layers.Conv2D):
            weights = oldlayer.get_weights()

            is_dw = isinstance(oldlayer, tf.keras.layers.DepthwiseConv2D) 
            if oldlayer.output.name in backward_prune: # !! se asume que la bn va justo despues de la conv
                backprune = backward_prune[oldlayer.output.name] # bool vector, comes from ahead
                if is_dw:
                    weights = [weights[n][:,:,backprune,:] for n in range(len(weights))]
                else:
                    weights = [weights[n][...,backprune] for n in range(len(weights))]
            else:
                backprune = np.full(weights[0].shape[-1], True) 
            if oldlayer.input.name in forward_prune and not is_dw: # for DWs it should never happen (?)
                if not oldlayer.use_bias and newlayer.use_bias:
                    weights.append(np.zeros(backprune.sum()))
                foreprune, betas = forward_prune[oldlayer.input.name] # foreprune comes from behind
                ker_sums = weights[0].sum(axis=(0,1)) # save kernels that go to pruned channels
                rescued_betas = (betas * ~foreprune).reshape((len(betas),1)) # to combine with betas
                weights[1] += (rescued_betas * ker_sums).sum(axis=0)
                weights[0] = weights[0][:,:,foreprune,:]
            newlayer.set_weights(weights)
            
    return new_model
        
