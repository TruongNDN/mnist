TL;DR : THE FIRST PART IS JUST ABOUT HOW TO RUN THE DATA
To download the data (already in .t7 format), run
```Shell
wget https://s3.amazonaws.com/torch7/data/mnist.t7.tgz
tar -xzf mnist.t7.tgz
rm mnist.t7.tgz
mv mnist.t7 data
```

To train MNIST, run
```Lua
th train.lua
```

We will now go deeper into the model and training procedure.

