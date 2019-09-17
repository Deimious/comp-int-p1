import numpy as np

learning_rate = 5
epochs = 1
error = 0

# in the fashion of [in1,in2,in3],[in1,in2,in3],...
inputs = np.array([[0.0, 0.0, 0.0], # Input #1
                   [0.0, 0.0, 1.0], # ...
                   [0.0, 1.0, 0.0],
                   [0.0, 1.0, 1.0],
                   [1.0, 0.0, 0.0],
                   [1.0, 0.0, 1.0],
                   [1.0, 1.0, 0.0],
                   [1.0, 1.0, 1.0]] # Input #8
)

# similar fashion to input. [out1,out2],[out1,out2],...
outputs = np.array([[1,0], # Output #1
                    [0,1], # ...
                    [0,1],
                    [1,0],
                    [0,1],
                    [1,0],
                    [1,0],
                    [1,0]] # Output #8
)

# weights for H1. First row is the connection from all of the inputs to neuron 1 in 
# the hidden layer. Second row is neuron 2, third is 3.
h1_weights = np.array([[0.1,0.2,0.3],
                       [0.1,0.1,0.1],
                       [0.3,0.3,0.3]]
)

# biases for hidden layer 1 in form: n1,n2,n3
h1_biases = np.array([[0.2],
                      [0.1],
                      [0.9]]
)

# weights for H2. First row is the connection from all of the inputs to neuron 2 in 
# the hidden layer. Second row is neuron 2, third is 3, fourth is 4.
h2_weights = np.array([[0.0,0.0,0.0],
                       [0.1,0.1,0.1],
                       [0.1,0.1,0.1],
                       [0.2,0.2,0.2]]
)

# biases for hidden layer 2 in form: n1,n2,n3,n4
h2_biases = np.array([[0.0],
                      [0.2],
                      [0.0],
                      [-0.1]]
)

# weights for all the h2 neurons to the neurons in the output layer (op)
op_weights = np.array([[1.5,1.2,1.0,0.0],
                       [0.0,0.8,0.1,0.0]]
)

if __name__ == "__main__":
    # print the matrices jusst to make sure everything's correct. It also looks like something is 
    # happening at this point, which in my frustration over my own stupidity makes me feel somewhat accomplished.
    #print("inputs:\n")
    #print(inputs); print("");
    #print("outputs:\n")
    #print(outputs); print("");
    #print("h1 weights:\n")
    #print(h1_weights); print("");
    #print("h1 biases:\n")
    #print(h1_biases); print("");
    #print("h2 weights:\n")
    #print(h2_weights); print("");
    #print("h2 biases:\n")
    #print(h2_biases); print("");


    for i in range(epochs):
        for index in range(8):
            h1 = np.tanh(np.dot(inputs[index,:],h1_weights) + np.transpose(h1_biases))
            h2 = np.tanh(np.dot(h1,np.transpose(h2_weights)) + np.transpose(h2_biases))
            o = np.tanh(np.dot(h2,np.transpose(op_weights)))

            cost = (0.5 * np.power(outputs[index] - o, 2.0));
            #print(cost)
            #print("h1 values:");print(h1);
            #print("h2 values:");print(h2);
            #print("output values:");print(o);
            delta1 = (-1.0) * (outputs[index] - o)
            # get the deriv of this idiot
            delta2 = np.tanh(o)
            combined = delta1 * delta2
            #print("delta1 for",index);print(delta1)
            #print("delta2 for",index);print(delta2)
            #print("delta1 * delta2: \n")
            #print(combined)
            delta_ow = np.ones((2,4))
            delta_ow = op_weights.T * combined
            delta3 = np.tanh(h2)

            delta_h2 = np.ones((3,4))
            delta_h2 = ((h2 + h2_biases.T) * delta3) * (combined.T * op_weights)

            delta4 = np.tanh(h1)
            delta_h1 = np.ones((3,3))
            #delta_h1 = ((h1 + h1_biases.T ) * delta4 * h2_weights) * (combined.T)  
            delta_h1 = (h1 + (h1_biases.T) * (delta4)) * (h2_weights)
            delta_h1 = np.delete(delta_h1,[0.0,0.0,0.0,0.0],axis=0)
            delta_h2 = np.vstack([delta_h2, [0.0,0.0,0.0,0.0]])
            #print("delta h1: ");print(delta_h1)
            #print("delta h2: ");print(delta_h2)
            #print("delta ow: ");print(delta_ow)

            h1_weights = h1_weights - learning_rate * delta_h1;
            h2_weights = h2_weights - learning_rate * delta_h2.T;
            op_weights = op_weights - learning_rate * delta_ow.T;
 

            #print("delta_h2: ");print(delta_h2)
            
            #print("delta_ow: ");print(delta_ow);
            
    #print("final h1 values:");print(h1);
    #print("final h2 values:");print(h2);
    #print("final output values:");print(o);
    print("h1 weights: ");print(h1_weights);
    print("h2 weights: ");print(h2_weights);
    print("o weights: ");print(h1_weights);
    print("h1 biases: ");print(h1_biases);
    print("h2 biases: ");print(h2_biases);
