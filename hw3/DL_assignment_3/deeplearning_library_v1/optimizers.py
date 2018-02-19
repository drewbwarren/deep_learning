import autograd.numpy as np
from autograd import value_and_grad, grad
from autograd.misc.flatten import flatten_func


        
    
# gradient descent function - inputs: g (input function), alpha (steplength parameter), max_its (maximum number of iterations), w (initialization)
def gradient_descent(g,alpha_choice,max_its,w, version=None, momentum=False, beta=0):
    # Handle optional arguments
    #if 'version' in kwargs:
    #    version = kwargs['version']
    #if 'beta' in kwargs:
    #    beta = kwargs['beta']
        
    # flatten the input function to more easily deal with costs that have layers of parameters
    g_flat, unflatten, w = flatten_func(g, w) # note here the output 'w' is also flattened

    # compute the gradient function of our input function - note this is a function too
    # that - when evaluated - returns both the gradient and function evaluations (remember
    # as discussed in Chapter 3 we always ge the function evaluation 'for free' when we use
    # an Automatic Differntiator to evaluate the gradient)
    gradient = value_and_grad(g_flat)

    # run the gradient descent loop
    weight_history = []      # container for weight history
    cost_history = []        # container for corresponding cost function history
    weight_history.append(unflatten(w))
    
    # for normalized
    best_w = w
    best_eval, _ = gradient(w)
    
    # for momentum
    z = np.zeros((np.shape(w)))
    
    alpha = 0
    for k in range(1,max_its+1):
        # check if diminishing steplength rule used
        if alpha_choice == 'diminishing':
            alpha = 1/float(k)
        else:
            alpha = alpha_choice

        # evaluate the gradient
        cost_eval,grad_eval = gradient(w)
                 
        if version == 'normalized':
            grad_norm = np.linalg.norm(grad_eval)
            if grad_norm == 0:
                grad_eval = 10**-6*np.sign(2*np.random.rand(len(w)) - 1)
                grad_norm = np.linalg.norm(grad_eval)
            grad_eval /= grad_norm
                 
        # take gradient descent step
        if momentum:
            z = beta*z + grad_eval
            w = w - alpha*z
        else:
            w = w - alpha*grad_eval
                 
        # Find best weight for normalized
        if version == 'normalized':
            test_eval, _ = gradient(w)
            if test_eval < best_eval:
                best_eval = test_eval
                best_w = w
        
        weight_history.append(unflatten(w))
        cost_history.append(cost_eval)

    # collect final weights
    if version == 'normalized':
        weight_history.append(unflatten(best_w))
        cost_history.append(g_flat(best_w))
    else:                 
        weight_history.append(unflatten(w))
        # compute final cost function value via g itself (since we aren't computing 
        # the gradient at the final step we don't get the final cost function value 
        # via the Automatic Differentiatoor) 
        cost_history.append(g_flat(w))  
    
    return weight_history,cost_history
   
    
# gradient descent function - inputs: g (input function), alpha (steplength parameter), max_its (maximum number of iterations), w (initialization)
def normalized_gradient_descent(g,alpha,max_its,w):
    # flatten the input function to more easily deal with costs that have layers of parameters
    g_flat, unflatten, w = flatten_func(g, w) # note here the output 'w' is also flattened
    print(w)
       
    # compute the gradient of our input function - note this is a function too!
    gradient = value_and_grad(g_flat)

    # run the gradient descent loop
    best_w = w        # weight we return, should be the one providing lowest evaluation
    best_eval,_ = gradient(w)       # lowest evaluation yet
    weight_history = []      # container for weight history
    cost_history = []        # container for corresponding cost function history
        
    for k in range(max_its):
        # evaluate the gradient, compute its length
        cost_eval, grad_eval = gradient(w)            
        # split it up into the separate matrices for each layer
        grad_norm = np.linalg.norm(grad_eval)

        # check that magnitude of gradient is not too small, if yes pick a random direction to move
        if grad_norm == 0:
            # pick random direction and normalize to have unit legnth
            grad_eval = 10**-6*np.sign(2*np.random.rand(len(w)) - 1)
            grad_norm = np.linalg.norm(grad_eval)

        # do this for each matrix of weights
        grad_eval /= grad_norm
        
        # take gradient descent step
        w = w - alpha*grad_eval

        # return only the weight providing the lowest evaluation
        test_eval, _ = gradient(w)
        if test_eval < best_eval:
            best_eval = test_eval
            best_w = w
            
        print(k)
            
        weight_history.append(unflatten(w))
        cost_history.append(g_flat(w))
        
    weight_history.append(unflatten(best_w))
    cost_history.append(g_flat(best_w))
    return weight_history,cost_history

