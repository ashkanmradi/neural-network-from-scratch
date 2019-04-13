import numpy as np
from queue import Queue

class Graph:
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []

    def as_default(self):
        global _default_graph
        _default_graph = self




def layerLin(x, w, b):
    return add(matmul(x, w), b)

def layerRelu(x, w, b):
    return Relu(add(matmul(x, w), b))

def layerLoss(c, p):
    return negative(sigma(sigma(multiply(c, log(p)), axis=1)))



class Operation:

    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.consumers = []

        for input_node in input_nodes:
            input_node.consumers.append(self)
        _default_graph.operations.append(self)

    def forward(self):
        pass

    def gradient(self):
        pass

class add(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def forward(self, x_value, y_value):
        return x_value + y_value

    def gradient(self,op, grad):
        grad_wrt_a = grad
        grad_wrt_b = grad

        a = op.inputs[0]
        b = op.inputs[1]

        while np.ndim(grad_wrt_a) > len(a.shape):
            grad_wrt_a = np.sum(grad_wrt_a, axis=0)
        for axis, size in enumerate(a.shape):
            if size == 1:
                grad_wrt_a = np.sum(grad_wrt_a, axis=axis, keepdims=True)
        
        while np.ndim(grad_wrt_b) > len(b.shape):
            grad_wrt_b = np.sum(grad_wrt_b, axis=0)
        for axis, size in enumerate(b.shape):
            if size == 1:
                grad_wrt_b = np.sum(grad_wrt_b, axis=axis, keepdims=True)


        return [grad_wrt_a, grad_wrt_b]

class matmul(Operation):
    def __init__(self, a, b):
        super().__init__([a, b])
    def forward(self, a_value, b_value):
        return a_value.dot(b_value)
    def gradient(self,op, grad):
        A = op.inputs[0]
        B = op.inputs[1]
        return [grad.dot(B.T), A.T.dot(grad)]

class sigmoid(Operation):
    def __init__(self, a):
        super().__init__([a])

    def forward(self, a_value):
        return 1 / (1 + np.exp(-a_value))

    def gradient(self,op, grad):
        sigmoid = op.output
        return grad * sigmoid * (1 - sigmoid)


class softmax(Operation):
    def __init__(self, a):
        super().__init__([a])

    def forward(self, a_value):
        return np.exp(a_value) / np.sum(np.exp(a_value), axis=1)[:, None]


    def gradient(self,op, grad):
        softmax = op.output
        return (grad - np.reshape(np.sum(grad * softmax, 1),[-1, 1])) * softmax

class log(Operation):
    def __init__(self, x):
        super().__init__([x])

    def forward(self, x_value):
        return np.log(x_value)

    def gradient(self,op, grad):
        x = op.inputs[0]
        return grad / x

class multiply(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def forward(self, x_value, y_value):
        return x_value * y_value

    def gradient(self,op, grad):
        A = op.inputs[0]
        B = op.inputs[1]
        return [grad * B, grad * A]

class sigma(Operation):
    def __init__(self, A, axis=None):
        super().__init__([A])
        self.axis = axis

    def forward(self, A_value):
        return np.sum(A_value, self.axis)

    def gradient(self,op, grad):
        A = op.inputs[0]
        output_shape = np.array(A.shape)
        output_shape[op.axis] = 1
        tile_scaling = A.shape // output_shape
        grad = np.reshape(grad, output_shape)
        return np.tile(grad, tile_scaling)



class negative(Operation):
    def __init__(self, x):
        super().__init__([x])

    def forward(self, x_value):
        return -x_value

    def gradient(self,op, grad):
        return -grad

class Relu(Operation):
    def __init__(self, x):
        super().__init__([x])

    def forward(self, x_value):
        return np.maximum(x_value, 0)

    def gradient(self,op, grad):
        x = op.inputs[0]
        x[x <= 0] = 0
        x[x > 0] = 1
        return -grad * x



class batch_normalize(Operation):
    def __init__(self, x,mean,std):
        super().__init__([x,mean,std])

    def forward(self, x_value, mean_value, std_value):
        return (x_value-mean_value) /std_value

    def gradient(op, grad):
        std = op.inputs[2]
        return grad/std



class placeholder:
    def __init__(self):
        self.consumers = []

        _default_graph.placeholders.append(self)

class Variable:
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.consumers = []

        _default_graph.variables.append(self)



class Session:
    def run(self, operation, feed_dict={}):

        #topological sort:
        nodes_postorder = topological_order(operation)

        for node in nodes_postorder:
            if type(node) == placeholder:
                node.output = feed_dict[node]

            elif type(node) == Variable:
                node.output = node.value

            else:
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.forward(*node.inputs)

            if type(node.output) == list:
                node.output = np.array(node.output)
        return operation.output


def topological_order(operation):
    nodes_postorder = []
    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder


class GradientDescentOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def minimize(self, loss):
        learning_rate = self.learning_rate

        class MinimizationOperation(Operation):
            def forward(self):
                grad_table = compute_gradients(loss)

                for node in grad_table:
                    if type(node) == Variable:
                        grad = grad_table[node]
                        node.value -= learning_rate * grad
        return MinimizationOperation()



Adamm={}
Adamv={}
class AdamOptimizer:
    def __init__(self, learning_rate, beta1, beta2, epsilon):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def minimize(self, loss):
        learning_rate = self.learning_rate
        beta1 = self.beta1
        beta2 = self.beta2
        epsilon = self.epsilon

        class MinimizationOperation(Operation):
            def forward(self):
                global grad_table
                grad_table = compute_gradients(loss)

                for node in grad_table:
                    if type(node) == Variable:
                        grad = grad_table[node]
                        if node not in Adamm:
                            Adamm[node]=0
                        if node not in Adamv:
                            Adamv[node]=0

                        Adamm[node]=beta1*Adamm[node] + (1-beta1)*grad
                        Adamv[node] = beta2 * Adamv[node] + (1 - beta2) * (grad**2)
                        mb = Adamm[node] / (1 - beta1 ** batchStep)
                        vb = Adamv[node] / (1 - beta2 ** batchStep)
                        node.value -= learning_rate * mb/(np.sqrt(vb)+epsilon)

        return MinimizationOperation()




rmspropCache={}
class RMSPropOptimizer:
    def __init__(self, learning_rate, decayRate, epsilon):
        self.learning_rate = learning_rate
        self.decayRate = decayRate
        self.epsilon = epsilon

    def minimize(self, loss):
        learning_rate = self.learning_rate
        decayRate = self.decayRate
        epsilon = self.epsilon

        class MinimizationOperation(Operation):
            def forward(self):
                global grad_table
                grad_table = compute_gradients(loss)

                for node in grad_table:
                    if type(node) == Variable:
                        grad = grad_table[node]
                        if node not in rmspropCache:
                            rmspropCache[node]=0

                        rmspropCache[node]=decayRate*rmspropCache[node] + (1-decayRate)*(grad**2)
                        node.value -= learning_rate * grad/(np.sqrt(rmspropCache[node])+epsilon)

        return MinimizationOperation()




def compute_gradients(loss):
    grad_table = {}
    grad_table[loss] = 1

    #BFS:
    visited = set()
    queue = Queue()
    visited.add(loss)
    queue.put(loss)

    while not queue.empty():
        node = queue.get()
        if node != loss:
            grad_table[node] = 0

            for consumer in node.consumers:
                lossgrad_wrt_consumer_output = grad_table[consumer]
                #### code 2:
                lossgrads_wrt_consumer_inputs = consumer.gradient(consumer, lossgrad_wrt_consumer_output)

                if len(consumer.input_nodes) == 1:
                    grad_table[node] += lossgrads_wrt_consumer_inputs
                else:
                    node_index_in_consumer_inputs = consumer.input_nodes.index(node)
                    lossgrad_wrt_node = lossgrads_wrt_consumer_inputs[node_index_in_consumer_inputs]
                    grad_table[node] += lossgrad_wrt_node

        if hasattr(node, "input_nodes"):
            for input_node in node.input_nodes:
                if not input_node in visited:
                    visited.add(input_node)
                    queue.put(input_node)

    return grad_table






import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, cmap = plt.cm.Blues):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           title=None,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    normalize = False
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



batchStep = 1
def setParams(loadFile=False,path="",epochNumPath=""):
    # Initialize weights randomly
    global W_1
    global b_1

    global W_2
    global b_2

    global W_3
    global b_3

    global W_4
    global b_4






    if (loadFile):
        import pickle
        b_1tmp = Variable()

        W_2tmp = Variable()
        b_2tmp = Variable()

        W_3tmp = Variable()
        b_3tmp = Variable()

        W_4tmp = Variable()
        b_4tmp = Variable()

        with open(path+'W1_Epoch_' + epochNumPath, 'rb') as f:
            W_1tmp = pickle.load(f)
            f.close()
        with open(path+'W2_Epoch_' + epochNumPath, 'rb') as f:
            W_2tmp = pickle.load(f)
            f.close()
        with open(path+'W3_Epoch_' + epochNumPath, 'rb') as f:
            W_3tmp = pickle.load(f)
            f.close()
        with open(path+'W4_Epoch_' + epochNumPath, 'rb') as f:
            W_4tmp = pickle.load(f)
            f.close()

        with open(path+'b1_Epoch_' + epochNumPath, 'rb') as f:
            b_1tmp = pickle.load(f)
            f.close()
        with open(path+'b2_Epoch_' + epochNumPath, 'rb') as f:
            b_2tmp = pickle.load(f)
            f.close()
        with open(path+'b3_Epoch_' + epochNumPath, 'rb') as f:
            b_3tmp = pickle.load(f)
            f.close()
        with open(path+'b4_Epoch_' + epochNumPath, 'rb') as f:
            b_4tmp = pickle.load(f)
            f.close()

        W_1 = Variable(W_1tmp.value)
        W_2 = Variable(W_2tmp.value)
        W_3 = Variable(W_3tmp.value)
        W_4 = Variable(W_4tmp.value)

        b_1 = Variable(b_1tmp.value)
        b_2 = Variable(b_2tmp.value)
        b_3 = Variable(b_3tmp.value)
        b_4 = Variable(b_4tmp.value)

    else:
        W_1 = Variable(np.random.randn(1024, 4000) / 100000)
        b_1 = Variable(np.random.randn(4000) / 100000)

        W_2 = Variable(np.random.randn(4000, 4000) / 1000000)
        b_2 = Variable(np.random.randn(4000) / 1000000)

        W_3 = Variable(np.random.randn(4000, 4000) / 1000000)
        b_3 = Variable(np.random.randn(4000) / 1000000)

        W_4 = Variable(np.random.randn(4000, 10) / 1000000)
        b_4 = Variable(np.random.randn(10) / 1000000)
    #batch_normalize_1_mean=Variable(np.zeros(1))
    #batch_normalize_1_std=Variable(np.ones(1))

def loadData():

    trainData=load_data.load.returnTrainDataSet()
    global trainDataX
    global trainDataC
    global testDataX
    global testDataC

    trainDataX=trainData[0]
    trainDataX = (trainDataX-np.mean(trainDataX,axis=0)) /np.std(trainDataX,axis=0)

    trainDataC=trainData[1]


    testData = load_data.load.returnTestDataSet()
    testDataX = testData[0]
    testDataX = (testDataX-np.mean(testDataX,axis=0)) /np.std(testDataX,axis=0)

    testDataC = testData[1]


def trainModel():
    global W_1
    global b_1

    global W_2
    global b_2

    global W_3
    global b_3

    global W_4
    global b_4
    session = Session()


    global batchStep
    X_axis = []
    Y_axis = []

    for step in range(1,epochsNum+1):
        start = 0
        end = start + batch_size

        for batch in range(int(len(trainDataX) / batch_size)):
            feed_dict = {
                X: trainDataX[start:end],
                c: trainDataC[start:end]
            }
            J_value = session.run(J, feed_dict)
            session.run(minimization_op, feed_dict)
            if (start/batch_size) % 1 ==0:
                print('epoch:', step,'/',epochsNum, ' batch:', int((end/batch_size)),'/',int((len(trainDataX)/batch_size)), '  loss:', J_value )
                X_axis.append(batchStep)
                Y_axis.append(J_value)
            start = end
            end = end + batch_size
            batchStep=batchStep+1

        if step % 1 == 0:
            testModel()
            print("----------- -----!Step:", step, " Loss:", J_value)

        XN_axis = np.array(X_axis)
        YN_axis = np.array(Y_axis)
        import matplotlib.pyplot as plt
        plt.xlabel('Epoch, Batch')
        plt.ylabel('Loss')
        plt.plot(XN_axis, YN_axis)
        plt.show()

        import pickle
        with open('F:\\New folder\W1_Epoch_'+str(step), 'wb') as f1:
            pickle.dump(W_1, f1)
            f1.close()
        with open('F:\\New folder\W2_Epoch_'+str(step), 'wb') as f2:
            pickle.dump(W_2, f2)
            f2.close()
        with open('F:\\New folder\W3_Epoch_'+str(step), 'wb') as f3:
            pickle.dump(W_3, f3)
            f3.close()
        with open('F:\\New folder\W4_Epoch_'+str(step), 'wb') as f4:
            pickle.dump(W_4, f4)
            f4.close()

        with open('F:\\New folder\\b1_Epoch_'+str(step), 'wb') as f5:
            pickle.dump(b_1, f5)
            f5.close()
        with open('F:\\New folder\\b2_Epoch_'+str(step), 'wb') as f6:
            pickle.dump(b_2, f6)
            f6.close()
        with open('F:\\New folder\\b3_Epoch_'+str(step), 'wb') as f7:
            pickle.dump(b_3, f7)
            f7.close()
        with open('F:\\New folder\\b4_Epoch_'+str(step), 'wb') as f8:
            pickle.dump(b_4, f8)
            f8.close()


def returnLoss(setparam=False):
    if (setparam):
        setParams()

    session = Session()
    start = 0
    end = start + batch_size

    test_feed_dict = {
        X: testDataX[start:end],
        c: testDataC[start:end]
    }
    loss = session.run(J, test_feed_dict)
    # print(temp.shape)
    return loss

def testModel():

    pred_class = np.empty((0, 10))
    start = 0
    end = start + batch_size
    session = Session()

    for batch in range(int(len(testDataX) / batch_size)):
        test_feed_dict = {
            X: testDataX[start:end],
            c: testDataC[start:end]
        }
        start = end
        end = start + batch_size
        temp = session.run(pred, test_feed_dict)
        #print(temp.shape)
        pred_class = np.append(pred_class, temp, axis=0)

    testDataCArgmax = np.argmax(testDataC, axis=1)
    # pred_class = np.array(pred_class)
    print(pred_class.shape)
    pred_classArgmax = np.argmax(pred_class, axis=1)
    cm = confusion_matrix(y_true=testDataCArgmax, y_pred=pred_classArgmax)
    # print(cm.shape)
    # f1_score=f1_score(true_class, pred_class, average=None)

    np.set_printoptions(precision=2)
    plot_confusion_matrix(cm)

    plt.show()
    acc=np.sum(testDataCArgmax == pred_classArgmax)/len(testDataCArgmax)
    print(acc)

from sklearn.metrics import confusion_matrix
import load_data.load

epochsNum=1
batch_size = 250


# Create a new graph
Graph().as_default()

X = placeholder()
c = placeholder()

setParams()
#setParams(loadFile=True,path="F:\\New folder\\",epochNumPath="11")

#batch_normalize_1=batch_normalize(hidden_1,batch_normalize_1_mean,batch_normalize_1_std)

hidden_1=layerRelu(X, W_1, b_1)
# hidden_2=layerLin(hidden_1, W_2, b_2)
hidden_3=layerRelu(hidden_1, W_3, b_3)
OutputLayer=layerLin(hidden_3, W_4, b_4)

pred = softmax(OutputLayer)

J = layerLoss(c, pred)


minimization_op = GradientDescentOptimizer(learning_rate=0.00005).minimize(J)
#minimization_op = RMSPropOptimizer(learning_rate=0.0002, decayRate=0.9,epsilon=1e-08).minimize(J)
#minimization_op = AdamOptimizer(learning_rate=0.0002, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(J)

loadData()
trainModel()
#testModel()
print(returnLoss())