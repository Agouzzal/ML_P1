import numpy as np

def loss_function_MSE(y,tx,w):
    residual = y-tx@w

    loss=(1/(2*tx.shape[0]))*np.transpose(residual)@residual

    return loss

def gradient(y,tx,w):
    residual = y-tx@w
    grad=-(1/tx.shape[0])*np.transpose(tx)@residual

    return grad


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):

    w=initial_w
    for i in range(max_iters):
        grad=gradient(y,tx,w)

        w=w-gamma*grad
    loss=loss_function_MSE(y,tx,w) 
    return w,loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    w=initial_w
    for i in range(max_iters):
        #Enonce demande batch de taille 1
        #On va chosir une ligne au hasard pour chaqeu iteration
        n=np.random.randint(tx.shape[0])
        grad=gradient(y[n],tx[n:n+1],w) # On slice tx, pour qu il reste une matrice et pas un vecteur
        w=w-gamma*grad
    loss=loss_function_MSE(y,tx,w)
    return w,loss
                

        



