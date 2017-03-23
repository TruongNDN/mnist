TL;DR : THE FIRST PART IS JUST ABOUT HOW TO RUN THE DATA
To download the data (already in .t7 format), run
```Shell
wget https://s3.amazonaws.com/torch7/data/mnist.t7.tgz
tar -xzf mnist.t7.tgz
rm mnist.t7.tgz
mv mnist.t7 data
```

To train MNIST, run
```Shell
th train.lua
```

If you use Lua JIT, be careful with the internal memory limit (2GB).
You might consider Torchnet for a more efficient way to load data.
We will now go deeper into the model and training procedure.

1. Loading data
2. Preparation (including setting up the model and other things)
3. Training
4. Evaluation

