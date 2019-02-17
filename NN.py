import numpy as np
import matplotlib.pyplot as plt
import csv

class MLP(object):

    def __init__(self,datapath=None,model_path=None,hidden_dims=(500,900),n_hidden=2,batch_size=32,lr=0.0001,feature_size=784,epoch=10,mode='train',weight_func='glorot',activ_func='relu',input_scaling=False,input_normalize=False):

        super(MLP, self).__init__()

        self.model_path = model_path
        self.tr, self.v, self.te = np.load(datapath)


        self.mode = mode
        self.activ_func = activ_func
        self.weight_func = weight_func
        self.batch_size = batch_size
        self.lr = lr
        self.epoch = epoch
        self.input_scaling = input_scaling
        self.input_normalize = input_normalize

        self.classes = [0,1,2,3,4,5,6,7,8,9]
        self.hidden_dims=hidden_dims
        self.L = n_hidden+2
        self.dims=(feature_size,)+self.hidden_dims+(len(self.classes),)


        self.weights = {}
        self.biais = {}
        self.computed_vals = {}
        self.grads={}
        
        self.tracking_train_loss = []
        self.tracking_valid_loss = []
        
        self.param = [self.hidden_dims[0],self.hidden_dims[1],self.lr,self.batch_size,self.activ_func]
        

    def mini_batch(self,data_type):

        input = []
        labels = []

        if data_type =='train':
            for i in range(0,len(self.tr[0]),self.batch_size):
                input.append(np.transpose(self.tr[0][i:min(i+self.batch_size,len(self.tr[0]))]))
                labels.append(np.transpose(self.tr[1][i:min(i+self.batch_size,len(self.tr[0]))]))

        elif data_type == 'valid':
            for i in range(0,len(self.v[0]),self.batch_size):
                input.append(np.transpose(self.v[0][i:min(i+self.batch_size,len(self.v[0]))]))
                labels.append(np.transpose(self.v[1][i:min(i+self.batch_size,len(self.v[0]))]))

        elif data_type == 'test' :
            for i in range(0,len(self.te[0]),self.batch_size):
                input.append(np.transpose(self.te[0][i:min(i+self.batch_size,len(self.te[0]))]))
                labels.append(np.transpose(self.te[1][i:min(i+self.batch_size,len(self.te[0]))]))

        else :
            assert False
        
        
        return input, labels #label batch dim = 32, input batch dim = 784x32

    
    def initialize_biais(self):
        
        for k in range(1,self.L):
                self.biais['b'+str(k)] = np.zeros((self.dims[k],1)) #dim b = hk x 1
    
    def initialize_weights(self):

        if self.weight_func == 'zero' :
            for k in range(1,self.L):
                self.weights['w'+str(k)] = np.zeros((self.dims[k], self.dims[k-1])) #dim w = hk x hk-1

        elif self.weight_func == 'normal' :
            for k in range(1,self.L):
                self.weights['w'+str(k)] = np.random.normal(np.zeros((self.dims[k], self.dims[k-1])), np.ones((self.dims[k], self.dims[k-1])))

        elif self.weight_func == 'glorot' :
            for k in range(1,self.L):
                self.weights['w'+str(k)] = np.random.uniform(-(6./(self.dims[k]+self.dims[k-1]))**(1./2.), (6./(self.dims[k]+self.dims[k-1]))**(1./2.),(self.dims[k], self.dims[k-1]))

        else :
            assert False

    def forward(self,input_batch):
        
        if self.input_scaling :
            input_batch = input_batch/255
        
        if self.input_normalize :
            input_batch = input_batch-255
        
        self.computed_vals['ac0'] = input_batch #input dim = 784 x 32

        for k in range(1,self.L-1):

            self.computed_vals['pre_ac'+str(k)] = np.dot(self.weights['w'+str(k)], self.computed_vals['ac'+str(k-1)]) + self.biais['b'+str(k)] #dim = hk x 32
            self.computed_vals['ac'+str(k)] = self.activation(self.computed_vals['pre_ac'+str(k)]) #dim hkx32


        self.computed_vals['pre_ac'+str(self.L-1)] = np.dot(self.weights['w'+str(self.L-1)], self.computed_vals['ac'+str(self.L-2)]) + self.biais['b'+str(self.L-1)] #dim = hout x 32
        self.computed_vals['ac'+str(self.L-1)] = self.softmax(self.computed_vals['pre_ac'+str(self.L-1)]) #dim = hout x 32


    def activation(self,input):

        if self.activ_func == 'relu':
            return (input>0)*input
        
        if self.activ_func == 'sigmoid':
            return np.exp(input)/(1+np.exp(input))
        
        if self.activ_func == 'tanh':
            return (np.exp(2*input)-1)/(np.exp(2*input)+1)


    def loss(self, out, labels):

        #out dim = 10x32
        #label dim = 1x32

        loss=0
        for i,v in enumerate(labels):
            loss -= np.log(out[v][i])

        return loss/len(labels)


    def softmax(self,input):

        out=[]
        for img in np.transpose(input) :
            out.append(np.exp(img)/np.sum(np.exp(img)))

        return np.transpose(np.array(out))


    def backward(self,labels):

        self.compute_grad_pre_ac_out(labels)

        for k in reversed(range(1,self.L-1)):
            self.compute_grad_hidden_ac(k)
            self.compute_grad_hidden_pre_ac(k)
            
        for k in range(1,self.L):
            self.compute_grad_weights(k)


    def compute_grad_pre_ac_out(self, labels):

        #d cross-entropy(softmax) / d pre_ac3
        
        
        grad = np.zeros((self.dims[self.L-1], len(labels))) #dim 10x32

        for i,v in enumerate(labels):
            grad[v][i] = 1

        self.grads['dpre_ac'+str(self.L-1)] = -(grad-self.computed_vals['ac'+str(self.L-1)]) #dim 10x32


    def compute_grad_hidden_ac(self, k):

        self.grads['dac'+str(k)] = np.dot(np.transpose(self.weights['w'+str(k+1)]),self.grads['dpre_ac'+str(k+1)]) #dim hkx32


    def compute_grad_hidden_pre_ac(self, k):

        self.grads['dpre_ac'+str(k)] = self.grads['dac'+str(k)] * self.grad_activation(self.computed_vals['pre_ac'+str(k)]) #dim hkx32

        

    def grad_activation(self,input):

        if self.activ_func == 'relu':
            return input>0
        
        if self.activ_func == 'sigmoid':
            return np.exp(input)/(1+np.exp(input)) * (1. - np.exp(input)/(1+np.exp(input)))
        
        if self.activ_func == 'tanh':
            return 1-(((np.exp(2*input)-1)/(np.exp(2*input)+1))**2)


    def compute_grad_weights(self,k):

        self.grads['dw'+str(k)] = np.dot(self.grads['dpre_ac'+str(k)], np.transpose(self.computed_vals['ac'+str(k-1)])) #dim hk x hk-1

    def update(self):

        """
        Class method that performs the gradient update
        :param self:
        :param weights: The weights to update
        :param grads: The gradients of the weights
        :param lr: The learning rate
        :return: The update for the weights
        """


        for k in range(1,self.L):
            self.weights['w'+str(k)] -= self.lr * self.grads['dw'+str(k)]
            self.biais['b'+str(k)] = self.biais['b'+str(k)] - self.lr * np.sum(self.grads['dpre_ac'+str(k)],axis=1,keepdims=True)

    
    def approx_grad(self,layer):
        """
        Class method that approximate the gradient of the loss function
        with the finite difference method performed on one sample
        :param layer: the layer in which we want to approxiamte some weight gradients
        """
        
        N=[]
        max_diff=[]
        
        #We use the first exemple of the training for the approximation
        #and perform forward and backward to compute the true gradients
        
        x = self.tr[0][0]
        x=np.reshape(x,(784,1))
        label = [self.tr[1][0]]
        self.forward(x)
        self.backward(label)
        
        
        # We compute the finite differences with several values of N
        
        for i in range(6):
            for k in range(1,6):
                
                # each value of N is added to the final list
                
                N.append(k*(10**i))
                
                # epsilon is a matrix of the same shape as the weight matrix
                # of which we approximate the gradients
                # We initilize it as a zeros matrix
                
                epsilon = np.zeros(self.weights['w'+str(layer)].shape)
                
                finite_diff=[]
                grads=[]
                
                # we perform the approximation for the 10 first values of 
                # the weight matrix
                
                for j in range(10):
                    
                    # at the position of the gradient we want to approxiamte
                    # epsilon take a very small value calulated with N
                    
                    epsilon[0][j] = 1./(N[-1])
                    
                    # we add/subtract epsilon to the weight matrix
                    # and then perform forward propagation
                    # to finally compute the finite difference
                    
                    self.weights['w'+str(layer)]+=epsilon
                    self.forward(x)
                    out = self.computed_vals['ac'+str(self.L-1)]
                    loss1 = self.loss(out,label)
                    self.weights['w'+str(layer)]-=(2*epsilon)
                    self.forward(x)
                    out = self.computed_vals['ac'+str(self.L-1)]
                    loss2 = self.loss(out,label)
                    
                    finite_diff.append((loss1-loss2)/(2./N[-1]))
                    
                    # we get also the corresponding true gradients
                    grads.append(self.grads['dw'+str(layer)][0][j])
                    
                    # we set the weight matrix back to its initial values
                    self.weights['w'+str(layer)]+=epsilon

                finite_diff = np.array(finite_diff)
                grads = np.array(grads)
                
                # we only keep the maximum diffrence btw the true gradient
                # and the finite difference for the 10 first weight elements 
                max_diff.append(max(np.absolute(finite_diff-grads)))
        
        print(N)
        print(max_diff)
        
        # plot the finite differences function of log(N)
        plt.plot(np.log(N),max_diff,'bo')
        plt.show
        return
    
    def train(self):
        
        self.initialize_biais()
        self.initialize_weights()
        current_epoch = 0        
        
        
        while current_epoch <= self.epoch :
            
            print('Epoque '+ str(current_epoch))
            train_i = 0
            valid_i = 0
            train_loss = 0
            valid_loss = 0
            acc = 0
            
            input_batch, labels_batch = self.mini_batch('train')
            for input,labels in zip(input_batch,labels_batch) :
                self.forward(input)
                out = self.computed_vals['ac'+str(self.L-1)]
                pred = [np.argmax(prob) for prob in np.transpose(out)] 
                loss = self.loss(out,labels)
                train_loss += loss
                if train_i%100 == 0 :
                    print('Loss train : '+str(loss))
                self.backward(labels)
                self.update()
                self.grads = {}
                self.computed_vals = {}
                train_i += 1
            
            print('\n')
            
            input_batch, labels_batch = self.mini_batch('valid')
            for input,labels in zip(input_batch,labels_batch) :
                self.forward(input)
                out = self.computed_vals['ac'+str(self.L-1)]
                pred = [np.argmax(prob) for prob in np.transpose(out)]  
                loss = self.loss(out,labels)
                acc += sum([1 if labels[i] == pred[i] else 0 for i in range(len(labels))])/float(len(labels))
                valid_loss += loss
                if valid_i%100 == 0 :
                    print('Loss valid : '+str(loss))
                self.grads = {}
                self.computed_vals = {}
                valid_i += 1
                
            self.tracking_train_loss.append(train_loss/train_i)
            self.tracking_valid_loss.append(valid_loss/valid_i)
            if current_epoch >= 2 :
                if self.tracking_valid_loss[-1] > self.tracking_valid_loss[-2] :
                    self.lr = self.lr / 2
            
            acc = float(acc)/valid_i
            
            print('\n')
            print('Accuracy : '+str(acc))
            
            current_epoch += 1
            
            print('\n')
        
        
        print(self.tracking_train_loss)
        print(self.tracking_valid_loss)
        plt.plot(self.tracking_train_loss, '-r', label='training')
        plt.plot(self.tracking_valid_loss, '-b', label='validation')
        plt.legend()
        plt.show()
        
        self.param.append(acc)

        with open('./hyperparameters_search.csv', 'a') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(self.param)

        csvFile.close()
