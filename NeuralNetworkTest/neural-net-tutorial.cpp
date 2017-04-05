
class Net
{
public:
	Net(topology);
	void feedForward(input);
	void backProp(target);
	void getResults(results) const;
private:
};

int main()
{
	Net myNet(topology);

	myNet.feedForward(input);//Training
	myNet.backProp(target);//Training
	myNet.getResults(results);
}