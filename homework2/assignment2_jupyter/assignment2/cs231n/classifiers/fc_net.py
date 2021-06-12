from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        X = np.reshape(X, (X.shape[0], -1))
        N, D = X.shape

        w1 = self.params['W1']
        b1 = self.params['b1']
        w2 = self.params['W2']
        b2 = self.params['b2']
        
        temp1, cache1 = affine_forward(X, w1, b1)
        temp2, cache2 = relu_forward(temp1)
        scores, cache3 = affine_forward(temp2,w2,b2)
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dscores = softmax_loss(scores,y)
        loss += self.reg*np.sum(w1*w1)*0.5 + self.reg*np.sum(w2*w2)*0.5
        d_temp2, dw2, db2 = affine_backward(dscores, cache3)        
        d_temp1 = relu_backward(d_temp2, cache2)
        dx, dw1, db1 = affine_backward(d_temp1, cache1)
        grads['W1'] = dw1 + 2*self.reg*w1*0.5
        grads['b1'] = db1
        grads['W2'] = dw2 + 2*self.reg*w2*0.5
        grads['b2'] = db2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        for i in range(1,self.num_layers):
          if i==1:
                self.params['W%s' %(i)] = weight_scale * np.random.randn(input_dim, hidden_dims[i-1])
                self.params['b%s' %(i)] = np.zeros(hidden_dims[i-1])
          else:
                self.params['W%s' %(i)] = weight_scale * np.random.randn(hidden_dims[i-2], hidden_dims[i-1])
                self.params['b%s' %(i)] = np.zeros(hidden_dims[i-1])
          if (self.normalization=="batchnorm")or(self.normalization=="layernorm"):
                self.params['gamma%s' %(i)] = np.ones(hidden_dims[i-1])
                self.params['beta%s' %(i)] = np.zeros(hidden_dims[i-1])
                

        self.params['W%s' %(self.num_layers)] = weight_scale * np.random.randn(hidden_dims[self.num_layers-2], num_classes)
        self.params['b%s' %(self.num_layers)] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        X = np.reshape(X, (X.shape[0], -1))
        vector = X
        history = []
        for i in range(1,self.num_layers):
              if self.normalization=="batchnorm":
                    temp1, cache1 = affine_forward(vector,self.params['W%s' %(i)], self.params['b%s' %(i)])
                    temp_batch, cache_batch = batchnorm_forward(temp1, self.params['gamma%s' %(i)], self.params['beta%s' %(i)], self.bn_params[i-1])
                    temp2, cache2 = relu_forward(temp_batch)
                    history.append((temp1, cache1))
                    history.append((temp_batch, cache_batch))
                    history.append((temp2, cache2))
                    vector = temp2
              if self.normalization=="layernorm":
                    temp1, cache1 = affine_forward(vector,self.params['W%s' %(i)], self.params['b%s' %(i)])
                    temp_layer, cache_layer = layernorm_forward(temp1, self.params['gamma%s' %(i)], self.params['beta%s' %(i)], self.bn_params[i-1])
                    temp2, cache2 = relu_forward(temp_layer)
                    history.append((temp1, cache1))
                    history.append((temp_layer, cache_layer))
                    history.append((temp2, cache2))
                    vector = temp2
              else:
                    temp1, cache1 = affine_forward(vector,self.params['W%s' %(i)], self.params['b%s' %(i)])
                    temp2, cache2 = relu_forward(temp1)
                    history.append((temp1, cache1))
                    history.append((temp2, cache2))
                    vector = temp2
        temp1, cache1 = affine_forward(vector,self.params['W%s' %(self.num_layers)], self.params['b%s' %(self.num_layers)])
        scores = temp1
        history.append((temp1, cache1))


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        
        loss, dscores = softmax_loss(scores, y)
        for i in range(1,self.num_layers+1):
              loss += 0.5 * self.reg * np.sum(self.params['W%s' %(i)]*self.params['W%s' %(i)])
        
        # 梯度向后传递
        dout = dscores
        for i in range(self.num_layers,0,-1):
              if self.normalization=="batchnorm":
                    if i!=1:
                          (temp1, cache1) = history.pop() # 取正交化的结果
                          (temp2, cache2) = history.pop() # 取激活函数的结果
                          (temp_batch, cache_batch) = history.pop() # 取批归一化的结果
                    else:
                          (temp1, cache1) = history.pop() # 取正交化的结果
                    if i!=1:
                          d_temp, grads['W%s' %(i)], grads['b%s' %(i)] = affine_backward(dout, cache1)
                          d_temp = relu_backward(d_temp, cache2)
                          d_temp, grads['gamma%s'%(i-1)], grads['beta%s' %(i-1)] = batchnorm_backward(d_temp, cache_batch)
                    else:
                          d_temp, grads['W%s' %(i)], grads['b%s' %(i)] = affine_backward(dout, cache1)
              if self.normalization=="layernorm":
                    if i!=1:
                          (temp1, cache1) = history.pop() # 取正交化的结果
                          (temp2, cache2) = history.pop() # 取激活函数的结果
                          (temp_layer, cache_layer) = history.pop() # 取批归一化的结果
                    else:
                          (temp1, cache1) = history.pop() # 取正交化的结果
                    if i!=1:
                          d_temp, grads['W%s' %(i)], grads['b%s' %(i)] = affine_backward(dout, cache1)
                          d_temp = relu_backward(d_temp, cache2)
                          d_temp, grads['gamma%s'%(i-1)], grads['beta%s' %(i-1)] = layernorm_backward(d_temp, cache_layer)
                    else:
                          d_temp, grads['W%s' %(i)], grads['b%s' %(i)] = affine_backward(dout, cache1)
              else:
                    if i!=1:
                          (temp1, cache1) = history.pop() # 取正交化的结果
                          (temp2, cache2) = history.pop() # 取激活函数的结果
                    else:
                          (temp1, cache1) = history.pop() # 取正交化的结果
                    
                    if i!=1:
                          d_temp, grads['W%s' %(i)], grads['b%s' %(i)] = affine_backward(dout, cache1)
                          d_temp = relu_backward(d_temp, cache2)
                    else:
                          d_temp, grads['W%s' %(i)], grads['b%s' %(i)] = affine_backward(dout, cache1)
              

              dout = d_temp
        
        for i in range(1,self.num_layers+1):
              grads['W%s' %(i)] += (0.5*2) * self.reg * self.params['W%s' %(i)]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def batchnorm_forward(x, gamma, beta, bn_param):

        mode = bn_param["mode"]
        eps = bn_param.get("eps", 1e-5)
        momentum = bn_param.get("momentum", 0.9)

        N, D = x.shape
        running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
        running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

        out, cache = None, None
        if mode == "train":
    
            sample_mean = running_mean
            sample_var = running_var
            running_mean = np.sum(x,axis=0)/N
            running_var = np.sqrt(np.sum((x-running_mean)**2,axis=0)/N)
            x_norm = (x-running_mean)/np.sqrt(running_var**2+eps)
            out = gamma*x_norm + beta
            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_var

            out1 = x
            out2 = x
            out3 = np.sum(out2,axis=0)/N
            out4 = out1 - out3
            out5 = out4
            out6 = out5**2
            out7 = np.sum(out6,axis=0)/N
            out8 = np.sqrt(out7 + eps)
            out9 = 1/out8
            out10 = out4 * out9
            out11 = gamma*out10
            out12 = out11 + beta
            out = out12

            cache = ((x,x_norm,gamma,beta,eps,running_mean,running_var),(x,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11,out12,gamma,beta,eps))

        elif mode == "test":

            running_mean = np.sum(x,axis=0)/N
            running_var = np.sqrt(np.sum((x-running_mean)**2,axis=0)/N)
            x_norm = (x-running_mean)/np.sqrt(running_var**2+eps)
            out = gamma*x_norm + beta

        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        bn_param["running_mean"] = running_mean
        bn_param["running_var"] = running_var

        return out, cache


    def batchnorm_backward(dout, cache):

        dx, dgamma, dbeta = None, None, None

        _,later = cache
        x,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11,out12,gamma,beta,eps = later
        N, D = x.shape
        dbeta = np.sum(dout,axis=0)
        dout11 = dout
        dgamma = np.sum(out10*dout11,axis=0)
        dout10 = dout11*gamma
        dout9 = np.sum(dout10*out4,axis=0)
        dout8 = -1/(out8**2) * dout9
        dout7 = 1/(2*np.sqrt(out7+eps))*dout8
        dout6 = dout7*np.ones_like(x)/N
        dout5 = dout6*2*out5
        dout4 = dout10*out9
        dout1 = dout4+dout5
        dout3 = -np.sum(dout4+dout5,axis=0)
        dout2 = dout3*np.ones_like(x)/N
        dx = dout1+dout2

        return dx, dgamma, dbeta