import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        # Compute dot product between weights and the input.
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        # Per the spec, if dot product is non-negative, return 1.
        score = nn.as_scalar(self.run(x))
        if score < 0:
            return -1 
        return 1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        while True:
            # Keep training until we don't fail on the input data.
            failure_encountered = False 

            # Iterate over input data. Not really sure what to specify for batch size so we're using 1.
            for x, y in dataset.iterate_once(1):
                # Get prediction using current weights.
                predicted = self.get_prediction(x)

                # If it's wrong, we gotta modify our weights.
                if predicted != nn.as_scalar(y):
                    # Indicate that it was wrong so we loop again, also.
                    failure_encountered = True 

                    # Update the weights using the provided update function. The "direction" is x. I picked x
                    # bc it has the same dimension as the weights, and I'm not sure what else we'd pick here...
                    self.get_weights().update(x, nn.as_scalar(y))
            
            # If we didn't mess up, then we can return.
            if not failure_encountered:
                return 

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.dim0 = 100
        # Can easily modify dim1 by changing divisor. Four and two seem to work fine.
        self.dim1 = int(self.dim0 / 4)

        self.w0 = nn.Parameter(1,self.dim0) 
        self.b0 = nn.Parameter(1,self.dim0)
        self.w1 = nn.Parameter(self.dim0, self.dim1)
        self.b1 = nn.Parameter(1, self.dim1) #nn.Parameter(1,1)
        self.w2 = nn.Parameter(self.dim1, 1)
        self.b2 = nn.Parameter(1,1)

        self.network = [self.w0, self.b0, self.w1, self.b1, self.w2, self.b2]

        self.learning_rate = 0.05
        self.batch_size = 50
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # Input is (batch_size x input_features) and (input_features x output_features).
        y0 = nn.Linear(x, self.w0)              # Output is (batch_size x input_features).
        y1 = nn.ReLU(nn.AddBias(y0, self.b0))   # Output is (batch_size x num_features).
        y2 = nn.Linear(y1, self.w1)             # Output is (batch_size x input_features).    
        y3 = nn.ReLU(nn.AddBias(y2, self.b1))   # Output is (batch_size x num_features).
        y4 = nn.Linear(y3, self.w2)
        y5 = nn.AddBias(y4, self.b2)
        return y5

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # We are supposed to call self.run(x) here (loss won't update in the GUI otherwise).
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        loss = 9999999
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                # Apparently we're supposed to pass 'x' here and call run from self.get_loss()...
                loss = self.get_loss(x, y)

                # Get the gradients.
                gradients = nn.gradients(loss, [self.w0, self.b0, self.w1, self.b1, self.w2, self.b2])

                self.w0.update(gradients[0], -self.learning_rate)
                self.b0.update(gradients[1], -self.learning_rate)
                self.w1.update(gradients[2], -self.learning_rate)
                self.b1.update(gradients[3], -self.learning_rate)
                self.w2.update(gradients[4], -self.learning_rate)
                self.b2.update(gradients[5], -self.learning_rate)
            
            # Make sure it gets below 0.02 by checking for 0.01. In testing though, this gets several orders of magnitude lower than 0.01.
            if nn.as_scalar(loss) < 0.01:
                return
        

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.dim0 = 784 # Cannot change.
        # Can easily modify dim1 by changing divisor. Four and two seem to work fine.
        self.dim1 = 392
        self.dim2 = 392
        self.final_dim = 10

        self.w0 = nn.Parameter(self.dim0, self.dim1) 
        self.b0 = nn.Parameter(1, self.dim1)
        self.w1 = nn.Parameter(self.dim1, self.dim2)
        self.b1 = nn.Parameter(1, self.dim2) #nn.Parameter(1,1)
        self.w2 = nn.Parameter(self.dim2, self.final_dim)
        self.b2 = nn.Parameter(1,self.final_dim)

        self.network = [self.w0, self.b0, self.w1, self.b1, self.w2, self.b2]

        self.learning_rate = 0.5
        self.batch_size = 100
        
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        y0 = nn.Linear(x, self.w0)              # Input is (batch_size x 784) and (784, self.dim1). Output is (batch_size, self.dim1).
        
        # The input to AddBias is (batch_size x num_features) and (1 x num_features), where num_features = self.dim1.
        # The output is (batch_size x num_features), where num_features = self.dim1. 
        y1 = nn.ReLU(nn.AddBias(y0, self.b0))  

        # Then, the input to Linear is (batch_size x input_features) and (input_features x output_features), 
        # meaning output is (batch_size x, output_features). In this case, the output will be (batch_size x self.dim2).
        y2 = nn.Linear(y1, self.w1)                

        # Input to add bias is (batch_size x self.dim2) and (1 x self.dim2). Output is (batch_size x self.dim2).
        y3 = nn.ReLU(nn.AddBias(y2, self.b1)) 

        # Input to linear is (batch_size x self.dim2) and (self.dim2 x 10). Output is (batch_size x 10).
        y4 = nn.Linear(y3, self.w2)
        y5 = nn.AddBias(y4, self.b2)
        return y5        

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        accuracy = 0.0
        epoch = 0
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                # Apparently we're supposed to pass 'x' here and call run from self.get_loss()...
                loss = self.get_loss(x, y)

                # Get the gradients.
                gradients = nn.gradients(loss, [self.w0, self.b0, self.w1, self.b1, self.w2, self.b2])

                self.w0.update(gradients[0], -self.learning_rate)
                self.b0.update(gradients[1], -self.learning_rate)
                self.w1.update(gradients[2], -self.learning_rate)
                self.b1.update(gradients[3], -self.learning_rate)
                self.w2.update(gradients[4], -self.learning_rate)
                self.b2.update(gradients[5], -self.learning_rate)
            
            accuracy = dataset.get_validation_accuracy()
            print("Accuracy = %f" % accuracy)
            # Make sure it gets below 0.02 by checking for 0.01. In testing though, this gets several orders of magnitude lower than 0.01.
            if accuracy >= 0.97:
                return        

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
