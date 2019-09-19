import numpy as np

# how fast the network should learn
learning_rate = 0.1

# only do one iteration of the network
epochs = 1

# init error
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

# desired output of [output neuron 1, output neuron 2] for each input
outputs = np.array([[1,0], # Output #1
                    [0,1], # ...
                    [0,1],
                    [1,0],
                    [0,1],
                    [1,0],
                    [1,0],
                    [1,0]] # Output #8
)

# weights from input neurons to h1 neurons
h1_weights = np.array([[0.1,0.2,0.3],  # weights attached to neuron 1
                       [0.1,0.1,0.1],  # weights attached to neuron 2
                       [0.3,0.3,0.3]]  # weights attached to neuron 3
)

# biases for hidden layer 1 in form: n1,n2,n3
h1_biases = np.array([[0.2],  # bias for n1
                      [0.1],  # bias for n2
                      [0.9]]  # bias for n3
)

# weights for H2. First row is the connection from all of the inputs to neuron 2 in 
# the hidden layer. Second row is neuron 2, third is 3, fourth is 4.
h2_weights = np.array([[0.0,0.0,0.0],  # weights attached to neuron 1
                       [0.1,0.1,0.1],  # weights attached to neuron 2
                       [0.1,0.1,0.1],  # weights attached to neuron 3
                       [0.2,0.2,0.2]]  # weights attached to neuron 4
)

# biases for hidden layer 2 in form: n1,n2,n3,n4
h2_biases = np.array([[0.0],  # bias for n1
                      [0.2],  # bias for n2
                      [0.0],  # bias for n3
                      [-0.1]] # bias for n4
)

# weights for all the h2 neurons to the neurons in the output layer (op)
op_weights = np.array([[1.5,1.2,1.0,0.0], # weights attached to neuron 1
                       [0.0,0.8,0.1,0.0]] # weights attached to neuron 2
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
            # forward pass
            # h1 holds values for the neurons in h1 after weights have been applied to inputs
            h1 = np.tanh(np.dot(inputs[index,:],h1_weights) + np.transpose(h1_biases))  # 1x3 matrix

            # h2 holds values for the neurons in h2 after weights have been applied to h1
            h2 = np.tanh(np.dot(h1,np.transpose(h2_weights)) + np.transpose(h2_biases)) # 1x4 matrix

            # o holds values for the neurons in o after weights have been applied to h2
            o = np.dot(h2,np.transpose(op_weights))                                     # 1x2 matrix

            # begin backpropogation

            # calculate the errors of the output
            error = (outputs[index] - o)
            #print("error: ");print(error);

            # compute the cost of these errors with SSE
            cost_ow = 0.5 * np.power(error[0,0] + error[0,1], 2.0)
            #print("cost_ow: ");print(cost_ow);

            delta_ow = np.ones((1,8))   # array of values to hold how much to change the op_weights by

            for w in range(2):
                for q in range(4):
                    sum_of_ow_weights = np.sum(op_weights)
                    #print("sum of op weights: ");print(sum_of_weights);
                    delta_ow[0,w] = ((op_weights[w,q])/sum_of_ow_weights) * error[0,w]

            #print("delta_ow: ");print(delta_ow)

            delta_hb2 = np.ones((1,4))  # array of values to hold how much to change the h2_biases by

            for b in range(4):
                sum_of_h2_biases = np.sum(h2_biases)
                #print("sum of h2 biases: ");print(sum_of_biases)
                delta_hb2[0,b] = ((h2_biases[b]/sum_of_h2_biases) * ((delta_ow[0,b] + delta_ow[0,b+1])/2));
                
            #print("delta_hb2: ");print(delta_hb2);

            delta_hw2 = np.ones((1,12)) # array of values to hold how much to change the h2_weights by

            i = 0
            for w in range(4):
                for q in range(3):
                    sum_of_h2_weights = np.sum(h2_weights)
                    #print("sum of h2 weights: ");print(sum_of_h2_weights)
                    delta_hw2[0,i] = (h2_weights[w,q]/sum_of_h2_weights) * ((delta_ow[0,(w+q)] + delta_ow[0,(w+q+1)]))
                    #print("delta_hw2: ");print(delta_hw2)
                    i += 1

            delta_hb1 = np.ones((1,3))  # array of values to hold how much to change the h1_biases by

            i = 0
            for b in range(3):
                sum_of_h1_biases = np.sum(h1_biases)
                #print("sum of h1 biases: ");print(sum_of_h1_biases)
                delta_hb1[0,b] = ((h1_biases[b]/sum_of_h1_biases) * ((delta_hw2[0,0+i] + delta_hw2[0,1+i] + delta_hw2[0,2+i] + delta_hw2[0,3+i])))
                #print("delta_hb1: ");print(delta_hb1)
                i += 4

            delta_hw1 = np.ones((1,9))  # array of values to hold how much to change the h1_weights by
           
            i = 0
            for w in range(3):
                for q in range(3):
                    sum_of_h1_weights = np.sum(h1_weights)
                    #print("sum of h1 weights: ");print(sum_of_h1_weights)
                    delta_hw1[0,i] = ((h1_weights[w,q]/sum_of_h1_biases) * delta_hw2[0,i+w])
                    i += 1


            for w in range(3):
                for q in range(3):
                    h1_weights[w,q] = h1_weights[w,q] + (learning_rate * delta_hw2[0,i+w])

            for b in range(3):
                h1_biases[b,0] = h1_biases[b,0] + (learning_rate * delta_hb1[0,b])

            i = 0
            for w in range(4):
                for q in range(3):
                    h2_weights[w,q] = h2_weights[w,q] + (learning_rate * delta_hw2[0,i])
                    i += 1

            for b in range(4):
                h2_biases[b,0] = h2_biases[b,0] + (learning_rate * delta_hb2[0,b])
            
            i = 0
            for w in range(2):
                for q in range(4):
                    op_weights[w,q] = op_weights[w,q] + (learning_rate * delta_ow[0,i])
                    i += 1
            #h1_weights = h1_weights + (learning_rate * delta_hw1)
            #h2_biases = h2_biases + (learning_rate * delta_hb2)
            #h2_weights = h1_weights + (learning_rate * delta_hw2)
            #op_weights = op_biases + (learning_rate * delta_ow)



            # calculate cost function for output & biases
            # adjust op_weights to match how to most efficiently reduce the cost
            # calculate cost function for h2 & biases
            # adjust h2_weights to match how to most efficiently reduce the cost
            # calculate cost function for h1 & biases
            # adjust h1_weights to match how to  most efficiently reduce the cost

            #print("cost: ");print(cost);
            #print("h1 values:");print(h1);
            #print("h2 values:");print(h2);
            #print("output values:");print(o);
            #delta1 = (-1.0) * (outputs[index] - o)
            # get the deriv of this idiot
            #delta2 = np.tanh(o)
            #combined = delta1 * delta2
            #print("delta1 for",index);print(delta1)
            #print("delta2 for",index);print(delta2)
            #print("delta1 * delta2: \n")
            #print(combined)
            #delta_ow = np.ones((2,4))
            #delta_ow = op_weights.T * combined
            #delta3 = np.tanh(h2)

            #delta_h2 = np.ones((3,4))
            #delta_h2 = ((h2 + h2_biases.T) * delta3) * (combined.T * op_weights)

            #delta4 = np.tanh(h1)
            #delta_h1 = np.ones((3,3))
            #delta_h1 = ((h1 + h1_biases.T ) * delta4 * h2_weights) * (combined.T)  
            #delta_h1 = (h1 + (h1_biases.T) * (delta4)) * (h2_weights)
            #delta_h1 = np.delete(delta_h1,[0.0,0.0,0.0,0.0],axis=0)
            #delta_h2 = np.vstack([delta_h2, [0.0,0.0,0.0,0.0]])
            #print("delta h1: ");print(delta_h1)
            #print("delta h2: ");print(delta_h2)
            #print("delta ow: ");print(delta_ow)

            #h1_weights = h1_weights - learning_rate * delta_h1;
            #h2_weights = h2_weights - learning_rate * delta_h2.T;
            #op_weights = op_weights - learning_rate * delta_ow.T;
 

            #print("delta_h2: ");print(delta_h2)
            
            #print("delta_ow: ");print(delta_ow);
            
    #print("final h1 values:");print(h1);
    #print("final h2 values:");print(h2);
    #print("final output values:");print(o);
    print("h1 weights: ");print(h1_weights);
    print("h2 weights: ");print(h2_weights);
    print("o weights: ");print(op_weights);
    print("h1 biases: ");print(h1_biases);
    print("h2 biases: ");print(h2_biases);
