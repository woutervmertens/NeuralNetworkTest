//Credits: Vinh Nguyen

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron;

typedef std::vector<Neuron> Layer;

//**********CLASS NEURON**********

class Neuron
{
public: 
	Neuron(unsigned numOutputs, unsigned m_myIndex);
	void setOutput(double val) { m_Output = val; }
	double getOutput(void) const { return m_Output; }
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double target);
	void calcHiddenGradients(const Layer & nextLayer);
	void updateInputWeights(Layer &prevLayer);
private:
	static double eta; // [0.0..1.0] overall net training rate
	static double alpha; // [0.0..n] multiplier of last weight change (momentum)
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double sumDOW(const Layer &nextLayer) const;
	double m_Output;
	std::vector<Connection> m_OutputWeights;
	unsigned m_myIndex;
	double m_gradient;
	
};

double Neuron::eta = 0.15; // overall net learning rate
double Neuron::alpha = 0.5; //momentum

void Neuron::updateInputWeights(Layer &prevLayer)
{
	//The weights to be updated are in the Connection container in the neurons in the preceding layer
	for(unsigned n = 0; n< prevLayer.size(); ++n)
	{
		Neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_OutputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight =
			//Individual input, magnified by the gradient and the train rate
			eta //overall net learning rate
			* neuron.getOutput()
			* m_gradient
			//also add momentum = a fraction of the previous delta weight
			+ alpha //momentum
			* oldDeltaWeight;
		neuron.m_OutputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_OutputWeights[m_myIndex].weight += newDeltaWeight;
	}
}

double Neuron::sumDOW(const Layer& nextLayer) const
{
	double sum = 0.0;

	//Sum our contributions of the errors at the nodes we feed
	for (unsigned n = 0; n < nextLayer.size() -1; ++n)
	{
		sum += m_OutputWeights[n].weight * nextLayer[n].m_gradient;
	}
	return sum;
}

void Neuron::calcHiddenGradients(const Layer & nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_Output);
}

void Neuron::calcOutputGradients(double target)
{
	double delta = target - m_Output;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_Output);
}

void Neuron::feedForward(const Layer &prevLayer)
{
	double sum = 0.0;

	//Sum the previous layer's outputs
	//Include the bias node from the previous layer

	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].getOutput() * prevLayer[n].m_OutputWeights[m_myIndex].weight;
	}
	m_Output = Neuron::transferFunction(sum);
}

double Neuron::transferFunction(double x)
{
	//tanh - output range [-1.0,1.0]
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
	//tanh derivative
	return 1.0 - x * x;
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
	for (unsigned c = 0; c < numOutputs; ++c)
	{
		m_OutputWeights.push_back(Connection());
		m_OutputWeights.back().weight = randomWeight();
	}
	m_myIndex = myIndex;
}


//**********CLASS NET***********
class Net
{
public:
	Net(const std::vector<unsigned> &topology);
	void feedForward(const std::vector<double> &input);//no need to pass by value and only reads
	void backProp(const std::vector<double> &target);
	void getResults(std::vector<double> &results) const;
private:
	std::vector<Layer> m_Layers; //m_Layers[layerNum][neuronNum]
	double m_error;
	double m_recentAverageError;
	double m_recentAverageSmoothingFactor;
};

void Net::getResults(std::vector<double> &results) const
{
	results.clear();
	for (unsigned n = 0; n < m_Layers.back().size() - 1; ++n)
		results.push_back(m_Layers.back()[n].getOutput());
}

void Net::feedForward(const std::vector<double> &input)
{
	assert(input.size() == m_Layers[0].size() - 1);

	//Assign (latch) the input values into the input neurons
	for (unsigned i = 0; i < input.size(); ++i)
	{
		m_Layers[0][i].setOutput(input[i]);
	}

	//Forward propagate
	for (unsigned layerNum = 1; layerNum < m_Layers.size(); ++layerNum)
	{
		Layer &prevLayer = m_Layers[layerNum - 1]; //& creates pointer
		for (unsigned n = 0; n < m_Layers[layerNum].size() - 1; ++n)
		{
			m_Layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

void Net::backProp(const std::vector<double>& target)
{
	//Calculate overall net error (RMS of output neuron errors)(RMS = Root Mean Square Error)
	Layer &outputLayer = m_Layers.back();
	m_error = 0.0;
	for (unsigned n = 0; n < outputLayer.size(); ++n)
	{
		double delta = target[n] - outputLayer[n].getOutput();
		m_error += delta * delta;
	}
	m_error /= outputLayer.size() - 1;//get average error squared
	m_error = sqrt(m_error); //RMS

	//Implement a recent average measurement:
	m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);

	//Calculate output layer gradients
	for (unsigned n = 0; n < outputLayer.size(); ++n)
	{
		outputLayer[n].calcOutputGradients(target[n]);
	}

	//Calculate gradients on hidden layers
	for(unsigned layerNum = m_Layers.size() - 2; layerNum > 0; --layerNum)
	{
		Layer & hiddenLayer = m_Layers[layerNum];
		Layer & nextLayer = m_Layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	//For all layer from outputs to first hidden layer
	//Update connection weights

	for(unsigned layerNum = m_Layers.size() - 1; layerNum > 0; --layerNum)
	{
		Layer &layer = m_Layers[layerNum];
		Layer & prevLayer = m_Layers[layerNum - 1];

		for(unsigned n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}

Net::Net(const std::vector<unsigned> &topology)
{
	unsigned numLayers = topology.size();
	for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
	{
		m_Layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		//We have a new Layer
		//Fill it with Neurons
		//Add a bias Neuron
		for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
		{
			m_Layers.back().push_back(Neuron(numOutputs, neuronNum));
			std::cout << "Made a Neuron!" << std::endl;
		}

		//Force the bias node's output value to 1.0
		m_Layers.back().back().setOutput(1.0);
	}
}

int main()
{
	//e.g., {3,2,1}
	std::vector<unsigned> topology;
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(1);
	Net myNet(topology);

	std::vector<double> input;
	myNet.feedForward(input);//Training

	std::vector<double> target;
	myNet.backProp(target);//Training

	std::vector<double> results;
	myNet.getResults(results);
}